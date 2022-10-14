import os
import time
import argparse
import numpy as np
import pandas as pd
from datetime import datetime
from attrdict import AttrDict

import loadSCM
import loadData
import loadModel

import pickle
from pprint import pprint

from scatter import *

import new.experiments as experiments
import new.train as train
import new.utils as utils
import new.global_vars as global_vars
from new.classes.memoize import Memoize



import shap
from explainer import Explainer



from random import seed
#RANDOM_SEED = 54321
#seed(RANDOM_SEED) # set the random seed so that the random permutations can be reproduced again
#np.random.seed(RANDOM_SEED)


@Memoize
def loadCausalModel(args, experiment_folder_name):
  return loadSCM.loadSCM(args.scm_class, experiment_folder_name)


@Memoize
def loadDataset(args, experiment_folder_name):
  # unused: experiment_folder_name
  if args.dataset_class == 'adult':
    return loadData.loadDataset(args.dataset_class, return_one_hot = False, load_from_cache = False, index_offset = 1)
  else:
    return loadData.loadDataset(args.dataset_class, return_one_hot = True, load_from_cache = False, meta_param = args.scm_class)


@Memoize
def loadClassifier(args, objs, experiment_folder_name):
  if args.classifier_class in global_vars.FAIR_MODELS:
    fair_nodes = utils.getTrainableNodesForFairModel(args, objs)
    # must have at least 1 endogenous node in the training set, otherwise we
    # cannot identify an action set (interventions do not affect exogenous nodes)
    if len([elem for elem in fair_nodes if 'x' in elem]) == 0:
      raise Exception(f'[INFO] No intervenable set of nodes founds to train `{args.classifier_class}`. Exiting.')
  else:
    fair_nodes = None

  return loadModel.loadModelForDataset(
    args.classifier_class,
    args.dataset_class,
    args.scm_class,
    args.num_train_samples,
    fair_nodes,
    args.fair_kernel_type,
    experiment_folder_name
  )

def hotTrainRecourseTypes(args, objs, recourse_types):
  start_time = time.time()
  print(f'\n' + '='*80 + '\n')
  print(f'[INFO] Hot-training ALIN, AKRR, GAUS, CVAE so they do not affect runtime...')
  training_handles = []
  if any(['alin' in elem for elem in recourse_types]): training_handles.append(train.trainRidge)
  if any(['akrr' in elem for elem in recourse_types]): training_handles.append(train.trainKernelRidge)
  if any(['gaus' in elem for elem in recourse_types]): training_handles.append(train.trainGP)
  if any(['cvae' in elem for elem in recourse_types]): training_handles.append(train.trainCVAE)
  for training_handle in training_handles:
    print()
    for node in objs.scm_obj.getTopologicalOrdering():
      parents = objs.scm_obj.getParentsForNode(node)
      if len(parents): # if not a root node
        training_handle(args, objs, node, parents)
  end_time = time.time()
  print(f'\n[INFO] Done (total warm-up time: {end_time - start_time}).')
  print(f'\n' + '='*80 + '\n')


if __name__ == "__main__":

  parser = argparse.ArgumentParser()

  #parser.add_argument('-s', '--scm_class', type=str, default='adult', help='Name of SCM to generate data using (see loadSCM.py)')
  parser.add_argument('-s', '--scm_class', type=str, default='german-credit', help='Name of SCM to generate data using (see loadSCM.py)')

  parser.add_argument('-d', '--dataset_class', type=str, default='synthetic', help='Name of dataset to train explanation model for: german, random, mortgage, twomoon')
  parser.add_argument('-c', '--classifier_class', type=str, default='mlp', help='Model class that will learn data: lr, mlp')
  parser.add_argument('-e', '--experiment', type=int, default=6, help='Which experiment to run (5,8=sanity; 6=table)')
  parser.add_argument('-p', '--process_id', type=str, default='0', help='When running parallel tests on the cluster, process_id guarantees (in addition to time stamped experiment folder) that experiments do not conflict.')

  #parser.add_argument('--experimental_setups', nargs = '+', type=str, default=['m1_cvae'])
  parser.add_argument('--experimental_setups', nargs = '+', type=str, default=['m0_true'])
  parser.add_argument('--norm_type', type=int, default=2)
  parser.add_argument('--lambda_lcb', type=float, default=2.5)
  parser.add_argument('--num_train_samples', type=int, default=250)
  parser.add_argument('--num_validation_samples', type=int, default=250)
  parser.add_argument('--num_display_samples', type=int, default=25)
  parser.add_argument('--num_fair_samples', type=int, default=10, help='number of negatively predicted samples selected from each of sensitive attribute groups (e.g., for `adult`: `num_train_samples` = 1500, `batch_number` = 0, `sample_count` = 1200, and `num_fair_samples` = 10).')
  parser.add_argument('--num_mc_samples', type=int, default=50)
  parser.add_argument('--debug_flag', type=bool, default=False)

  #parser.add_argument('--non_intervenable_nodes', nargs = '+', type=str, default=['x1', 'x2', 'x3', 'x4', 'x5'])  
  #parser.add_argument('--non_intervenable_nodes', nargs = '+', type=str, default=['x1', 'x2', 'x3', 'x4']) #adult
  parser.add_argument('--non_intervenable_nodes', nargs = '+', type=str, default=['x1', 'x2']) #german-credit
  #parser.add_argument('--non_intervenable_nodes', nargs = '+', type=str, default=[''])

  parser.add_argument('--sensitive_attribute_nodes', nargs = '+', type=str, default='')
  parser.add_argument('--fair_kernel_type', type=str, default='rbf')
  #parser.add_argument('--max_intervention_cardinality', type=int, default=2)
  parser.add_argument('--max_intervention_cardinality', type=int, default=100)

  parser.add_argument('--max_shap_intervention_cardinality', type=int, default=100)
  parser.add_argument('--attempts_per_sample', type=int, default=5)
  
  parser.add_argument('--optimization_approach', type=str, default='grad_descent')
  parser.add_argument('--grid_search_bins', type=int, default=10)
  parser.add_argument('--grad_descent_epochs', type=int, default=1000)
  parser.add_argument('--epsilon_boundary', type=int, default=0.10, help='we only consider instances that are negatively predicted and at least epsilon_boundary prob away from decision boundary (too restrictive = smaller `batch_number` possible w/ fixed `num_train_samples`).')
  parser.add_argument('--batch_number', type=int, default=0)
  parser.add_argument('--sample_count', type=int, default=50, help='number of negatively predicted samples chosen in this batch (must be less, and often ~50% of `num_train_samples`')
  #parser.add_argument('--sample_count', type=int, default=50, help='number of negatively predicted samples chosen in this batch (must be less, and often ~50% of `num_train_samples`')

  args = parser.parse_args()

  start_time = time.time()

  if not (args.dataset_class in {'synthetic', 'adult'}):
    raise Exception(f'{args.dataset_class} not supported.')

  # create experiment folder
  setup_name = \
    f'{args.scm_class}__{args.dataset_class}__{args.classifier_class}' + \
    f'__ntrain_{args.num_train_samples}' + \
    f'__nmc_{args.num_mc_samples}' + \
    f'__lambda_lcb_{args.lambda_lcb}' + \
    f'__opt_{args.optimization_approach}' + \
    f'__batch_{args.batch_number}' + \
    f'__count_{args.sample_count}' + \
    f'__pid{args.process_id}'
  experiment_folder_name = f"_experiments/{datetime.now().strftime('%Y.%m.%d_%H.%M.%S')}__{setup_name}"
  args.experiment_folder_name = experiment_folder_name
  os.mkdir(f'{experiment_folder_name}')

  # save all arguments to file
  args_file = open(f'{experiment_folder_name}/_args.txt','w')
  for arg in vars(args):
    print(arg, ':\t', getattr(args, arg), file = args_file)

  # only load once so shuffling order is the same
  scm_obj = loadCausalModel(args, experiment_folder_name)
  dataset_obj = loadDataset(args, experiment_folder_name)
  # IMPORTANT: for traversing, always ONLY use either:
  #     * objs.dataset_obj.getInputAttributeNames()
  #     * objs.scm_obj.getTopologicalOrdering()
  # DO NOT USE, e.g., for key in factual_instance.keys(), whose ordering may differ!
  # IMPORTANT: ordering may be [x3, x2, x1, x4, x5, x6] as is the case of the
  # 6-variable model.. this is OK b/c the dataset is generated as per this order
  # and consequently the model is trained as such as well (where the 1st feature
  # is x3 in the example above)
  if args.dataset_class != 'adult':
    assert \
      list(scm_obj.getTopologicalOrdering()) == \
      list(dataset_obj.getInputAttributeNames()) == \
      [elem for elem in dataset_obj.data_frame_kurz.columns if 'x' in elem] # endogenous variables must start with `x`

  # TODO (lowpri): add more assertions for columns of dataset matching the classifer?
  objs = AttrDict({
    'scm_obj': scm_obj,
    'dataset_obj': dataset_obj,
  })

  # for fair models, the classifier depends on the {dataset, scm}_objs
  classifier_obj = loadClassifier(args, objs, experiment_folder_name)
  objs['classifier_obj'] = classifier_obj

  # TODO (lowpri): describe scm_obj
  print(f'Describe original data:\n{utils.getOriginalDataFrame(objs, args.num_train_samples).describe()}')
  # TODO (lowpri): describe classifier_obj

  # if only visualizing
  if args.experiment == 0:
    args.num_display_samples = 150
    visualizeDatasetAndFixedModel(objs.dataset_obj, objs.classifier_obj, experiment_folder_name)
    quit()

  # setup
  factual_instances_dict = experiments.getNegativelyPredictedInstances(args, objs)

  objs['factual_instances_dict'] = factual_instances_dict

  experimental_setups = []
  for experimental_setup in args.experimental_setups:
    assert \
      experimental_setup in global_vars.ACCEPTABLE_POINT_RECOURSE or \
      experimental_setup in global_vars.ACCEPTABLE_DISTR_RECOURSE, \
      f'Experimental setup `{experimental_setup}` not recognized.'
    experimental_setups.append(
      [setup for setup in global_vars.EXPERIMENTAL_SETUPS if setup[0] == experimental_setup][0]
    )
  assert len(experimental_setups) > 0, 'Need at least 1 experimental setup.'

  ##############################################################################
  ##                              Perform some checks on the experimental setup
  ##############################################################################
  unique_labels_set = set(pd.unique(
    objs.dataset_obj.data_frame_kurz[
      objs.dataset_obj.getOutputAttributeNames()[0]
    ]
  ))
  run_any_prob_recourse = any([x[0] in global_vars.ACCEPTABLE_DISTR_RECOURSE for  x in experimental_setups])

  if run_any_prob_recourse:

    assert \
      'svm' not in args.classifier_class, \
      'SVM model cannot work with probabilistic recourse approach due to lack of predict_proba() method.'

    assert \
      unique_labels_set == {0,1}, \
      'prob recourse only with 0,1 labels (0.5 boundary)' # TODO (lowpri): convert to 0.0 for -1/+1 labels(and test)

  if args.classifier_class in global_vars.FAIR_MODELS and args.dataset_class != 'adult':
    assert \
      unique_labels_set == {-1,1}, \
      'fair classifiers only works with {-1/+1} labels' # TODO (lowpri): ok for SVMs other then iw_fair_svm?
      # TODO (lowpri): can we change this?? can sklearn svm and lr predict 0,1 instead of -1/+1??

  if args.scm_class == 'adult':

    assert \
      args.classifier_class not in {'unaware_svm', 'cw_fair_svm'}, \
      f'The `adult` dataset cannot work with `{args.classifier_class}` as it does not have intervenable nodes that are non-descendants of sensitive attributes.'

    if args.classifier_class in global_vars.FAIR_MODELS:
      for setup in experimental_setups:
        assert \
          setup[0] == 'm1_cvae', \
          'Only run the adult fair dataset using `m1_cvae` for categorical support'

  if args.classifier_class == 'iw_fair_svm':

    assert \
      len(args.sensitive_attribute_nodes) <= 1, \
      'iw_fair_svm only accepts 1 sensitive attribute' # TODO (lowpri): confirm

  if args.experiment == 5:

    assert \
      len(objs.dataset_obj.getInputAttributeNames()) == 3, \
      'Exp 5 is only designed for 3-variable SCMs'

  elif args.experiment == 6:

    assert \
      len(objs.dataset_obj.getInputAttributeNames()) >= 3, \
      'Exp 6 is only designed for 3+-variable SCMs'

  recourse_types = [experimental_setup[0] for experimental_setup in experimental_setups]
  hotTrainRecourseTypes(args, objs, recourse_types)

  if args.experiment == 5:
    experiments.runSubPlotSanity(args, objs, experiment_folder_name, experimental_setups, factual_instances_dict, recourse_types)
  
  elif args.experiment == 6:
    experiments.runBoxPlotSanity(args, objs, experiment_folder_name, experimental_setups, factual_instances_dict, recourse_types)
    
    results = experiments.runRecourseExperiment(args, objs, experiment_folder_name, experimental_setups, factual_instances_dict, recourse_types)
    end_time = time.time()
    results['total_runtime'] = np.around(end_time - start_time, 3)
    print(f'[INFO] Saving (overwriting) results...\t', end='')
    pprint(results, open(f'{experiment_folder_name}/_per_instance_results.txt', 'w'))
    print(f'done.')
  
  elif args.experiment == 8:
    experiments.runBoxPlotSanity(args, objs, experiment_folder_name, experimental_setups, factual_instances_dict, recourse_types)
  
  elif args.experiment == 9: # fair recourse
    experiments.runFairRecourseExperiment(args, objs, experiment_folder_name, experimental_setups, factual_instances_dict, recourse_types)
