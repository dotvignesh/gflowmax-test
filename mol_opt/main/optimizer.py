import os
import yaml
import random
import torch
import numpy as np
from rdkit import Chem
from rdkit.Chem import Draw
import tdc
from tdc.generation import MolGen
from main.utils.chem import *


class Objdict(dict):
    def __getattr__(self, name):
        if name in self:
            return self[name]
        else:
            raise AttributeError("No such attribute: " + name)

    def __setattr__(self, name, value):
        self[name] = value

    def __delattr__(self, name):
        if name in self:
            del self[name]
        else:
            raise AttributeError("No such attribute: " + name)


def top_auc(buffer, top_n, finish, freq_log, max_oracle_calls, normalize_by_calls=None):
    if len(buffer) == 0:
        return 0.0

    max_oracle_calls = max(1, int(max_oracle_calls))
    top_n = max(1, int(top_n))
    ordered_results = list(sorted(buffer.items(), key=lambda kv: kv[1][1], reverse=False))[:max_oracle_calls]
    observed_calls = len(ordered_results)
    if observed_calls == 0:
        return 0.0

    # Compute exact discrete AUC over oracle calls so metric is independent of logging frequency.
    import heapq

    auc_sum = 0.0
    top_sum = 0.0
    top_heap = []
    top_n_now = 0.0

    for _, (score, _) in ordered_results:
        score = float(score)
        if len(top_heap) < top_n:
            heapq.heappush(top_heap, score)
            top_sum += score
        elif score > top_heap[0]:
            top_sum += score - heapq.heapreplace(top_heap, score)
        top_n_now = top_sum / len(top_heap)
        auc_sum += top_n_now

    if finish and observed_calls < max_oracle_calls:
        auc_sum += (max_oracle_calls - observed_calls) * top_n_now

    if normalize_by_calls is None:
        denominator = max_oracle_calls
    else:
        denominator = max(1, int(normalize_by_calls))
    return auc_sum / denominator


class Oracle:
    def __init__(self, args=None, mol_buffer={}):
        self.name = None
        self.evaluator = None
        self.task_label = None
        if args is None:
            self.max_oracle_calls = 10000
            self.freq_log = 100
        else:
            self.args = args
            self.max_oracle_calls = args.max_oracle_calls
            self.freq_log = args.freq_log
        self.mol_buffer = mol_buffer
        self.sa_scorer = tdc.Oracle(name = 'SA')
        self.diversity_evaluator = tdc.Evaluator(name = 'Diversity')
        self.last_log = 0
        self._metric_warning_printed = False

    @property
    def budget(self):
        return self.max_oracle_calls

    def assign_evaluator(self, evaluator):
        self.evaluator = evaluator

    def sort_buffer(self):
        self.mol_buffer = dict(sorted(self.mol_buffer.items(), key=lambda kv: kv[1][0], reverse=True))

    def save_result(self, suffix=None):
        
        if suffix is None:
            output_file_path = os.path.join(self.args.output_dir, 'results.yaml')
        else:
            output_file_path = os.path.join(self.args.output_dir, 'results_' + suffix + '.yaml')

        self.sort_buffer()
        with open(output_file_path, 'w') as f:
            yaml.dump(self.mol_buffer, f, sort_keys=False)

    def log_intermediate(self, mols=None, scores=None, finish=False, normalize_auc_to_observed=False):

        if finish:
            temp_top100 = list(self.mol_buffer.items())[:100]
            smis = [item[0] for item in temp_top100]
            scores = [item[1][0] for item in temp_top100]
            n_calls = self.max_oracle_calls
        else:
            if mols is None and scores is None:
                if len(self.mol_buffer) <= self.max_oracle_calls:
                    # If not spefcified, log current top-100 mols in buffer
                    temp_top100 = list(self.mol_buffer.items())[:100]
                    smis = [item[0] for item in temp_top100]
                    scores = [item[1][0] for item in temp_top100]
                    n_calls = len(self.mol_buffer)
                else:
                    results = list(sorted(self.mol_buffer.items(), key=lambda kv: kv[1][1], reverse=False))[:self.max_oracle_calls]
                    temp_top100 = sorted(results, key=lambda kv: kv[1][0], reverse=True)[:100]
                    smis = [item[0] for item in temp_top100]
                    scores = [item[1][0] for item in temp_top100]
                    n_calls = self.max_oracle_calls
            else:
                # Otherwise, log the input moleucles
                smis = [Chem.MolToSmiles(m) for m in mols]
                n_calls = len(self.mol_buffer)
        
        # Uncomment this line if want to log top-10 moelucles figures, so as the best_mol key values.
        # temp_top10 = list(self.mol_buffer.items())[:10]

        avg_top1 = np.max(scores)
        avg_top10 = np.mean(sorted(scores, reverse=True)[:10])
        avg_top100 = np.mean(scores)
        try:
            avg_sa = np.mean(self.sa_scorer(smis))
            diversity_top100 = self.diversity_evaluator(smis)
        except Exception as exc:
            if not self._metric_warning_printed:
                print(f"[oracle] SA/diversity metrics unavailable, continuing without them: {exc}")
                self._metric_warning_printed = True
            avg_sa = -1.0
            diversity_top100 = -1.0
        
        print(f'{n_calls}/{self.max_oracle_calls} | '
                f'avg_top1: {avg_top1:.3f} | '
                f'avg_top10: {avg_top10:.3f} | '
                f'avg_top100: {avg_top100:.3f} | '
                f'avg_sa: {avg_sa:.3f} | '
                f'div: {diversity_top100:.3f}')

        # try:
        auc_normalize_calls = None
        if normalize_auc_to_observed:
            auc_normalize_calls = min(len(self.mol_buffer), self.max_oracle_calls)

        metrics = {
            "avg_top1": avg_top1, 
            "avg_top10": avg_top10, 
            "avg_top100": avg_top100, 
            "auc_top1": top_auc(
                self.mol_buffer,
                1,
                finish,
                self.freq_log,
                self.max_oracle_calls,
                normalize_by_calls=auc_normalize_calls,
            ),
            "auc_top10": top_auc(
                self.mol_buffer,
                10,
                finish,
                self.freq_log,
                self.max_oracle_calls,
                normalize_by_calls=auc_normalize_calls,
            ),
            "auc_top100": top_auc(
                self.mol_buffer,
                100,
                finish,
                self.freq_log,
                self.max_oracle_calls,
                normalize_by_calls=auc_normalize_calls,
            ),
            "avg_sa": avg_sa,
            "diversity_top100": diversity_top100,
            "n_oracle": n_calls,
        }
        if normalize_auc_to_observed:
            metrics["auc_norm_calls"] = max(1, auc_normalize_calls)

        print(metrics)


    def __len__(self):
        return len(self.mol_buffer) 

    def score_smi(self, smi):
        """
        Function to score one molecule

        Argguments:
            smi: One SMILES string represnets a moelcule.

        Return:
            score: a float represents the property of the molecule.
        """
        if len(self.mol_buffer) > self.max_oracle_calls:
            return 0
        if smi is None:
            return 0
        mol = Chem.MolFromSmiles(smi)
        if mol is None or len(smi) == 0:
            return 0
        else:
            smi = Chem.MolToSmiles(mol)
            if smi in self.mol_buffer:
                pass
            else:
                self.mol_buffer[smi] = [float(self.evaluator(smi)), len(self.mol_buffer)+1]
            return self.mol_buffer[smi][0]
    
    def __call__(self, smiles_lst):
        """
        Score
        """
        if type(smiles_lst) == list:
            score_list = []
            for smi in smiles_lst:
                score_list.append(self.score_smi(smi))
                if len(self.mol_buffer) % self.freq_log == 0 and len(self.mol_buffer) > self.last_log:
                    self.sort_buffer()
                    self.log_intermediate()
                    self.last_log = len(self.mol_buffer)
                    self.save_result(self.task_label)
        else:  ### a string of SMILES 
            score_list = self.score_smi(smiles_lst)
            if len(self.mol_buffer) % self.freq_log == 0 and len(self.mol_buffer) > self.last_log:
                self.sort_buffer()
                self.log_intermediate()
                self.last_log = len(self.mol_buffer)
                self.save_result(self.task_label)
        return score_list

    @property
    def finish(self):
        return len(self.mol_buffer) >= self.max_oracle_calls


class BaseOptimizer:

    def __init__(self, args=None):
        self.model_name = "Default"
        self.args = args
        self.n_jobs = args.n_jobs
        # self.pool = joblib.Parallel(n_jobs=self.n_jobs)
        self.smi_file = args.smi_file
        self.oracle = Oracle(args=self.args)
        if self.smi_file is not None:
            self.all_smiles = self.load_smiles_from_file(self.smi_file)
        else:
            data = MolGen(name = 'ZINC')
            self.all_smiles = data.get_data()['smiles'].tolist()
            
        self.sa_scorer = tdc.Oracle(name = 'SA')
        self.diversity_evaluator = tdc.Evaluator(name = 'Diversity')
        self._metric_warning_printed = False
        try:
            self.filter = tdc.chem_utils.oracle.filter.MolFilter(
                filters=['PAINS', 'SureChEMBL', 'Glaxo'],
                property_filters_flag=False,
            )
        except Exception as exc:
            # Keep optimization runnable when rd_filters/pip is unavailable.
            print(f"[optimizer] MolFilter unavailable, skipping pass-rate metrics: {exc}")
            self.filter = None

    # def load_smiles_from_file(self, file_name):
    #     with open(file_name) as f:
    #         return self.pool(delayed(canonicalize)(s.strip()) for s in f)
            
    def sanitize(self, mol_list):
        new_mol_list = []
        smiles_set = set()
        for mol in mol_list:
            if mol is not None:
                try:
                    smiles = Chem.MolToSmiles(mol)
                    if smiles is not None and smiles not in smiles_set:
                        smiles_set.add(smiles)
                        new_mol_list.append(mol)
                except ValueError:
                    print('bad smiles')
        return new_mol_list
        
    def sort_buffer(self):
        self.oracle.sort_buffer()
    
    def log_intermediate(self, mols=None, scores=None, finish=False, normalize_auc_to_observed=False):
        self.oracle.log_intermediate(
            mols=mols,
            scores=scores,
            finish=finish,
            normalize_auc_to_observed=normalize_auc_to_observed,
        )
    
    def log_result(self):

        print(f"Logging final results...")

        # import ipdb; ipdb.set_trace()

        log_num_oracles = [100, 500, 1000, 3000, 5000, 10000]
        assert len(self.mol_buffer) > 0 

        results = list(sorted(self.mol_buffer.items(), key=lambda kv: kv[1][1], reverse=False))
        if len(results) > 10000:
            results = results[:10000]
        
        results_all_level = []
        for n_o in log_num_oracles:
            results_all_level.append(sorted(results[:n_o], key=lambda kv: kv[1][0], reverse=True))
        

        # Log batch metrics at various oracle calls
        data = [[log_num_oracles[i]] + self._analyze_results(r) for i, r in enumerate(results_all_level)]
        columns = ["#Oracle", "avg_top100", "avg_top10", "avg_top1", "Diversity", "avg_SA", "%Pass", "Top-1 Pass"]
        
    def save_result(self, suffix=None):

        print(f"Saving molecules...")
        
        if suffix is None:
            output_file_path = os.path.join(self.args.output_dir, 'results.yaml')
        else:
            output_file_path = os.path.join(self.args.output_dir, 'results_' + suffix + '.yaml')

        self.sort_buffer()
        with open(output_file_path, 'w') as f:
            yaml.dump(self.mol_buffer, f, sort_keys=False)
    
    def _analyze_results(self, results):
        results = results[:100]
        scores_dict = {item[0]: item[1][0] for item in results}
        smis = [item[0] for item in results]
        scores = [item[1][0] for item in results]
        if self.filter is None:
            smis_pass = []
            top1_pass = -1
            pass_rate = -1.0
        else:
            smis_pass = self.filter(smis)
            if len(smis_pass) == 0:
                top1_pass = -1
            else:
                top1_pass = np.max([scores_dict[s] for s in smis_pass])
            pass_rate = float(len(smis_pass) / 100)
        try:
            diversity = self.diversity_evaluator(smis)
            avg_sa = np.mean(self.sa_scorer(smis))
        except Exception as exc:
            if not self._metric_warning_printed:
                print(f"[optimizer] SA/diversity metrics unavailable, reporting -1 values: {exc}")
                self._metric_warning_printed = True
            diversity = -1.0
            avg_sa = -1.0
        return [np.mean(scores), 
                np.mean(scores[:10]), 
                np.max(scores), 
                diversity, 
                avg_sa, 
                pass_rate, 
                top1_pass]

    def reset(self):
        del self.oracle
        self.oracle = Oracle(args=self.args)

    @property
    def mol_buffer(self):
        return self.oracle.mol_buffer

    @property
    def finish(self):
        return self.oracle.finish
        
    def _optimize(self, oracle, config):
        raise NotImplementedError
            
    def hparam_tune(self, oracles, hparam_space, hparam_default, count=5, num_runs=3, project="tune"):
        seeds = [0, 1, 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
        seeds = seeds[:num_runs]
        hparam_space["name"] = hparam_space["name"]
        
        def _func():
            avg_auc = 0
            for oracle in oracles:
                auc_top10s = []
                for seed in seeds:
                    np.random.seed(seed)
                    torch.manual_seed(seed)
                    random.seed(seed)
                    self._optimize(oracle, config)
                    auc_top10s.append(top_auc(self.oracle.mol_buffer, 10, True, self.oracle.freq_log, self.oracle.max_oracle_calls))
                    self.reset()
                avg_auc += np.mean(auc_top10s)
            print({"avg_auc": avg_auc})
            

    def optimize(self, oracle, config, seed=0, project="test"):

        np.random.seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)
        self.seed = seed 
        self.oracle.task_label = self.model_name + "_" + oracle.name + "_" + str(seed)
        self._optimize(oracle, config)
        if self.args.log_results:
            self.log_result()
        self.save_result(self.model_name + "_" + oracle.name + "_" + str(seed))
        self.reset()


    def production(self, oracle, config, num_runs=5, project="production"):
        seeds = [0, 1, 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
        # seeds = [23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
        if num_runs > len(seeds):
            raise ValueError(f"Current implementation only allows at most {len(seeds)} runs.")
        seeds = seeds[:num_runs]
        for seed in seeds:
            self.optimize(oracle, config, seed, project)
            self.reset()
