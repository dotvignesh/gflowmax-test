from __future__ import print_function

import argparse
import yaml
import os
import sys
sys.path.append(os.path.realpath(__file__))

# TDC on some versions expects rdkit.six, removed in newer RDKit builds.
try:
    import rdkit.six  # noqa: F401
except Exception:
    import six as _six
    sys.modules["rdkit.six"] = _six

from tdc import Oracle
from time import time 


def _configure_cuda(cuda_arg):
    """Configure CUDA visibility before importing torch-heavy method modules.

    Supported values:
      - auto: keep default environment
      - cpu: force CPU by hiding CUDA devices
      - <id> or <id0,id1,...>: expose only selected GPU ids
    """
    if cuda_arg is None:
        return
    value = str(cuda_arg).strip().lower()
    if value == "auto":
        return
    if value == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        return

    ids = [part.strip() for part in str(cuda_arg).split(",")]
    if not ids or any(not item.isdigit() for item in ids):
        raise ValueError(
            f"Invalid --cuda value: {cuda_arg}. Use one of: auto, cpu, 0, 1, 0,1"
        )
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(ids)


def main(method,oracle_name):
    start_time = time() 
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', default=method)#'reinvent')
    parser.add_argument('--smi_file', default=None)
    parser.add_argument('--config_default', default='hparams_default.yaml')
    parser.add_argument('--config_tune', default='hparams_tune.yaml')
    parser.add_argument('--pickle_directory', help='Directory containing pickle files with the distribution statistics', default=None)
    parser.add_argument('--n_jobs', type=int, default=-1)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--max_oracle_calls', type=int, default=10000)
    parser.add_argument('--freq_log', type=int, default=100)
    parser.add_argument('--n_runs', type=int, default=5)
    # parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--seed', type=int, nargs="+", default=[0])
    parser.add_argument('--task', type=str, default="simple", choices=["tune", "simple", "production"])
    oracle_list=["ASKCOS","IBM_RXN","GSK3B","JNK3","DRD2","SA","QED","LogP","Rediscovery","Celecoxib_Rediscovery","Rediscovery_Meta"]
    parser.add_argument('--oracles', nargs="+", default=[oracle_name]) ### default=["QED"]#JNK3   memory large
    parser.add_argument('--log_results', action='store_true')
    parser.add_argument('--log_code', action='store_true')
    parser.add_argument('--wandb', type=str, default="offline", choices=["online", "offline", "disabled"])
    parser.add_argument('--cuda', type=str, default="auto",
                        help="CUDA device selection: auto | cpu | <gpu_id> | <gpu_id0,gpu_id1,...>")
    args = parser.parse_args()

    os.environ["WANDB_MODE"] = args.wandb
    _configure_cuda(args.cuda)

    if not args.log_code:
        os.environ["WANDB_DISABLE_CODE"] = "false"

    args.method = args.method.lower()
    method_aliases = {
        "gflowmax-mdp": "gflowmax_mdp",
        "reinvent-gfm": "reinvent_gfm",
        "genetic-gfn": "genetic_gfn",
        "genetic-gfn-selfies": "genetic_gfn_selfies",
        "genetic-gfn-al": "genetic_gfn_al",
        "reinvent-ls-gfn": "reinvent_ls_gfn",
    }
    args.method = method_aliases.get(args.method, args.method)

    path_main = os.path.dirname(os.path.realpath(__file__))
    path_main = os.path.join(path_main, "main", args.method)

    sys.path.append(path_main)
    
    print(args.method)
    # Add method name here when adding new ones
    if args.method == 'screening':
        from main.screening.run import Exhaustive_Optimizer as Optimizer 
    elif args.method == 'molpal':
        from main.molpal.run import MolPAL_Optimizer as Optimizer
    elif args.method == 'graph_ga':
        from main.graph_ga.run import GB_GA_Optimizer as Optimizer
    elif args.method == 'smiles_ga':
        from main.smiles_ga.run import SMILES_GA_Optimizer as Optimizer
    elif args.method == "selfies_ga":
        from main.selfies_ga.run import SELFIES_GA_Optimizer as Optimizer
    elif args.method == "synnet":
        from main.synnet.run import SynNet_Optimizer as Optimizer
    elif args.method == 'hebo':
        from main.hebo.run import HEBO_Optimizer as Optimizer 
    elif args.method == 'graph_mcts':
        from main.graph_mcts.run import Graph_MCTS_Optimizer as Optimizer
    elif args.method == 'smiles_ahc':
        from main.smiles_ahc.run import AHC_Optimizer as Optimizer
    elif args.method == 'smiles_aug_mem':
        from main.smiles_aug_mem.run import AugmentedMemory_Optimizer as Optimizer 
    elif args.method == 'smiles_bar':
        from main.smiles_bar.run import BAR_Optimizer as Optimizer 
    elif args.method == "smiles_lstm_hc":
        from main.smiles_lstm_hc.run import SMILES_LSTM_HC_Optimizer as Optimizer
    elif args.method == 'selfies_lstm_hc':
        from main.selfies_lstm_hc.run import SELFIES_LSTM_HC_Optimizer as Optimizer
    elif args.method == 'dog_gen':
        from main.dog_gen.run import DoG_Gen_Optimizer as Optimizer
    elif args.method == 'gegl':
        from main.gegl.run import GEGL_Optimizer as Optimizer 
    elif args.method == 'boss':
        from main.boss.run import BOSS_Optimizer as Optimizer
    elif args.method == 'chembo':
        from main.chembo.run import ChemBOoptimizer as Optimizer 
    elif args.method == 'gpbo':
        from main.gpbo.run import GPBO_Optimizer as Optimizer
    elif args.method == 'stoned': 
        from main.stoned.run import Stoned_Optimizer as Optimizer
    elif args.method == "selfies_vae":
        from main.selfies_vae.run import SELFIES_VAEBO_Optimizer as Optimizer
    elif args.method == "smiles_vae":
        from main.smiles_vae.run import SMILES_VAEBO_Optimizer as Optimizer
    elif args.method == 'jt_vae':
        from main.jt_vae.run import JTVAE_BO_Optimizer as Optimizer
    elif args.method == 'dog_ae':
        from main.dog_ae.run import DoG_AE_Optimizer as Optimizer
    elif args.method == 'pasithea':
        from main.pasithea.run import Pasithea_Optimizer as Optimizer
    elif args.method == 'dst':
        from main.dst.run import DST_Optimizer as Optimizer        
    elif args.method == 'molgan':
        from main.molgan.run import MolGAN_Optimizer as Optimizer
    elif args.method == 'mars':
        from main.mars.run import MARS_Optimizer as Optimizer
    elif args.method == 'mimosa':
        from main.mimosa.run import MIMOSA_Optimizer as Optimizer
    elif args.method == 'gflownet':
        from main.gflownet.run import GFlowNet_Optimizer as Optimizer
    elif args.method == 'gflownet_al':
        from main.gflownet_al.run import GFlowNet_AL_Optimizer as Optimizer
    elif args.method == 'gflowmax':
        from main.gflowmax.run import GFlowMax_Optimizer as Optimizer
    elif args.method == 'gflowmax_mdp':
        from main.gflowmax_mdp.run import GFlowMaxMDP_Optimizer as Optimizer
    elif args.method == 'moldqn':
        from main.moldqn.run import MolDQN_Optimizer as Optimizer
    elif args.method == 'reinvent':
        from main.reinvent.run import REINVENT_Optimizer as Optimizer
    elif args.method == 'reinvent_transformer':
        from main.reinvent_transformer.run_transformer import REINVENT_Optimizer as Optimizer
    elif args.method == 'reinvent_selfies':
        from main.reinvent_selfies.run import REINVENT_SELFIES_Optimizer as Optimizer
    elif args.method == 'genetic_gfn':
        from main.genetic_gfn.run import Genetic_GFN_Optimizer as Optimizer
    elif args.method == 'genetic_gfn_selfies':
        from main.genetic_gfn_selfies.run import Genetic_GFN_SELFIES_Optimizer as Optimizer
    elif args.method == 'genetic_gfn_al':
        from main.genetic_gfn_al.run import Genetic_GFN_AL_Optimizer as Optimizer
    elif args.method == 'reinvent_ls_gfn':
        from main.genetic_gfn.run_ls_gfn import REINVENT_LS_GFN_Optimizer as Optimizer
        path_main = os.path.join(os.path.dirname(os.path.realpath(__file__)), "main", "genetic_gfn")
    elif args.method == 'reinvent_gfm':
        from main.reinvent_gfm.run import REINVENT_GFM_Optimizer as Optimizer
    elif args.method == 'graphinvent':
        from main.graphinvent.run import GraphInvent_Optimizer as Optimizer
    elif args.method == "rationale_rl":
        from main.rationale_rl.run import Rationale_RL_Optimizer as Optimizer
    else:
        raise ValueError("Unrecognized method name.")


    if args.output_dir is None:
        args.output_dir = os.path.join(path_main, "results")
    
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    if args.pickle_directory is None:
        args.pickle_directory = path_main

    if args.task != "tune":
    
        for oracle_name in args.oracles:

            print(f'Optimizing oracle function: {oracle_name}')

            try:
                config_default = yaml.safe_load(open(args.config_default))
            except:
                config_default = yaml.safe_load(open(os.path.join(path_main, args.config_default)))

            oracle = Oracle(name = oracle_name)
            optimizer = Optimizer(args=args)

            if args.task == "simple":
                # optimizer.optimize(oracle=oracle, config=config_default, seed=args.seed) 
                for seed in args.seed:
                    print('seed', seed)
                    optimizer.optimize(oracle=oracle, config=config_default, seed=seed)
            elif args.task == "production":
                optimizer.production(oracle=oracle, config=config_default, num_runs=args.n_runs)
            else:
                raise ValueError('Unrecognized task name, task should be in one of simple, tune and production.')

    elif args.task == "tune":

        print(f'Tuning hyper-parameters on tasks: {args.oracles}')

        try:
            config_default = yaml.safe_load(open(args.config_default))
        except:
            config_default = yaml.safe_load(open(os.path.join(path_main, args.config_default)))

        try:
            config_tune = yaml.safe_load(open(args.config_tune))
        except:
            config_tune = yaml.safe_load(open(os.path.join(path_main, args.config_tune)))

        oracles = [Oracle(name = oracle_name) for oracle_name in args.oracles]
        optimizer = Optimizer(args=args)
        
        optimizer.hparam_tune(oracles=oracles, hparam_space=config_tune, hparam_default=config_default, count=args.n_runs)

    else:
        raise ValueError('Unrecognized task name, task should be in one of simple, tune and production.')
    end_time = time()
    hours = (end_time - start_time) / 3600.0
    print('---- The whole process takes %.2f hours ----' % (hours))
    # print('If the program does not exit, press control+c.')


if __name__ == "__main__":
    for oracle_name in ["Albuterol_Similarity", "Amlodipine_MPO", 
                        "Celecoxib_Rediscovery", "Deco_Hop", "DRD2", 
                        "Fexofenadine_MPO", "GSK3B", "Isomers_C7H8N2O2",
                        "Isomers_C7H8N2O3", "Isomers_C9H10N2O2PF2Cl", "JNK3",
                        "Median 1", "Median 2", "Mestranol_Similarity", 
                        "Osimertinib_MPO", "Perindopril_MPO", "QED", "Ranolazine_MPO",
                        "Scaffold_Hop", "Sitagliptin_MPO", "Thiothixene_Rediscovery", 
                        "Troglitazone_Rediscovery", "Valsartan_Smarts", "Zaleplon_MPO"]: #["Zaleplon_MPO","Amlodipine_MPO","GSK3B"]:#["Thiothixene_Rediscovery","Troglitazone_Rediscovery","Valsartan_Smarts","Zaleplon_MPO","Amlodipine_MPO","GSK3B"]:
        for method in ["reinvent_transformer","reinvent"]:
            main(method,oracle_name)
