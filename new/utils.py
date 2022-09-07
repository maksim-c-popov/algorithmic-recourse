import numpy as np
import pandas as pd

import itertools
from new.classes.memoize import Memoize
from new.classes.instance import Instance
import new.sampling as sampling
import new.global_vars as global_vars


@Memoize
def getOriginalDataFrame(objs, num_samples, with_meta = False, with_label = False, balanced = True, data_split = 'train_and_test'):

  return objs.dataset_obj.getOriginalDataFrame(
    num_samples = num_samples,
    with_meta = with_meta,
    with_label = with_label,
    balanced = balanced,
    data_split = data_split
  )


def convertToOneHotWithPrespecifiedCategories(df, node, lower_bound, upper_bound):
  return pd.get_dummies(
    pd.Categorical(
      df[node],
      categories=list(range(
        int(lower_bound),
        int(upper_bound) + 1
      ))
    ),
    prefix=node
  )


def powerset(iterable):
  "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
  s = list(iterable)
  return itertools.chain.from_iterable(itertools.combinations(s, r) for r in range(len(s)+1))


def getConditionalString(node, parents):
  return f'p({node} | {", ".join(parents)})'


def getNoiseStringForNode(node):
  assert node[0] == 'x'
  return 'u' + node[1:]


def processDataFrameOrInstance(args, objs, obj, processing_type):
  # TODO (cat): add support for categorical data

  if processing_type == 'raw':
    return obj

  if isinstance(obj, Instance):
    raise NotImplementedError
    # iterate_over = obj.keys()
  elif isinstance(obj, pd.DataFrame):
    iterate_over = obj.columns
  else:
    raise Exception(f'Datatype `{obj.__class__}` not supported for processing.')

  obj = obj.copy() # so as not to change the underlying object
  for node in iterate_over:
    if 'u' in node:
      print(f'[WARNING] Skipping over processing of noise variable {node}.')
      continue
    # use objs.dataset_obj stats, not X_train (in case you use more samples later,
    # e.g., validation set for cvae
    tmp = objs.dataset_obj.data_frame_kurz.describe()[node]
    node_min = tmp['min']
    node_max = tmp['max']
    node_mean = tmp['mean']
    node_std = tmp['std']
    if processing_type == 'normalize':
      obj[node] = (obj[node] - node_min) / (node_max - node_min)
    elif processing_type == 'standardize':
      obj[node] = (obj[node] - node_mean) / node_std
    elif processing_type == 'mean_subtract':
      obj[node] = (obj[node] - node_mean)
  return obj


def deprocessDataFrameOrInstance(args, objs, obj, processing_type):
  # TODO (cat): add support for categorical data

  if processing_type == 'raw':
    return obj

  if isinstance(obj, Instance):
    raise NotImplementedError
    # iterate_over = obj.keys()
  elif isinstance(obj, pd.DataFrame):
    iterate_over = obj.columns
  else:
    raise Exception(f'Datatype `{obj.__class__}` not supported for processing.')

  obj = obj.copy() # so as not to change the underlying object
  for node in iterate_over:
    if 'u' in node:
      print(f'[WARNING] Skipping over processing of noise variable {node}.')
      continue
    # use objs.dataset_obj stats, not X_train (in case you use more samples later,
    # e.g., validation set for cvae
    tmp = objs.dataset_obj.data_frame_kurz.describe()[node]
    node_min = tmp['min']
    node_max = tmp['max']
    node_mean = tmp['mean']
    node_std = tmp['std']
    if processing_type == 'normalize':
      obj[node] = obj[node] * (node_max - node_min) + node_min
    elif processing_type == 'standardize':
      obj[node] = obj[node] * node_std + node_mean
    elif processing_type == 'mean_subtract':
      obj[node] = obj[node] + node_mean
  return obj


def getTrainableNodesForFairModel(args, objs):

  sensitive_attribute_nodes = args.sensitive_attribute_nodes
  non_sensitive_attribute_nodes = np.setdiff1d(
    objs.dataset_obj.getInputAttributeNames('kurz'),
    list(sensitive_attribute_nodes)
  )

  if len(sensitive_attribute_nodes):
    unaware_nodes = [list(objs.scm_obj.getNonDescendentsForNode(node)) for node in args.sensitive_attribute_nodes]
    if len(unaware_nodes) > 1:
      unaware_nodes = set(np.intersect1d(*unaware_nodes))
    else:
      unaware_nodes = unaware_nodes[0]
    aware_nodes = np.setdiff1d(
      objs.dataset_obj.getInputAttributeNames('kurz'),
      list(unaware_nodes)
    )
    aware_nodes_noise = [getNoiseStringForNode(node) for node in aware_nodes]
  else:
    unaware_nodes = objs.dataset_obj.getInputAttributeNames('kurz')
    aware_nodes = []
    aware_nodes_noise = []


  if args.classifier_class == 'vanilla_svm' or args.classifier_class == 'vanilla_lr' or args.classifier_class == 'vanilla_mlp':
    fair_endogenous_nodes = objs.dataset_obj.getInputAttributeNames('kurz')
    fair_exogenous_nodes = []

  elif args.classifier_class == 'nonsens_svm' or args.classifier_class == 'nonsens_lr' or args.classifier_class == 'nonsens_mlp':
    fair_endogenous_nodes = non_sensitive_attribute_nodes
    fair_exogenous_nodes = []

  elif args.classifier_class == 'unaware_svm' or args.classifier_class == 'unaware_lr' or args.classifier_class == 'unaware_mlp':
    fair_endogenous_nodes = unaware_nodes
    fair_exogenous_nodes = []

  elif args.classifier_class == 'cw_fair_svm' or args.classifier_class == 'cw_fair_lr' or args.classifier_class == 'cw_fair_mlp':
    fair_endogenous_nodes = unaware_nodes
    fair_exogenous_nodes = aware_nodes_noise

  elif args.classifier_class == 'iw_fair_svm':
    fair_endogenous_nodes = objs.dataset_obj.getInputAttributeNames('kurz')
    fair_exogenous_nodes = []

  # just to be safe (does happens sometimes) that columns are not ordered;
  # if not sorted, this will be a problem for iw-fair-train which assumes
  # that the first column is the sensitive attribute.
  fair_endogenous_nodes = np.sort(fair_endogenous_nodes)
  fair_exogenous_nodes = np.sort(fair_exogenous_nodes)

  return np.concatenate((fair_endogenous_nodes, fair_exogenous_nodes))


def computeCounterfactualInstance(args, objs, factual_instance_obj, action_set, recourse_type):

  # assert recourse_type in ACCEPTABLE_POINT_RECOURSE # TODO (highpri): uncomment after having ran adult experiments

  if not bool(action_set): # if action_set is empty, CFE = F
    return factual_instance_obj

  samples_df = sampling._sampleInnerLoop(args, objs, factual_instance_obj, action_set, recourse_type, 1)

  counter_instance_dict = samples_df.loc[0].to_dict() # only endogenous variables
  return Instance({
    **factual_instance_obj.dict('endogenous_and_exogenous'), # copy endogenous and exogenous variables
    **counter_instance_dict # overwrite endogenous variables
  }) 


def isPredictionOfInstanceInClass(args, objs, instance_obj, prediction_class):
  # get instance for trained model
  if args.classifier_class in global_vars.FAIR_MODELS:

    fair_nodes = getTrainableNodesForFairModel(args, objs)
    instance_dict = dict(zip(
      fair_nodes,
      [instance_obj.dict('endogenous_and_exogenous')[key] for key in fair_nodes]
    ))

  else:
    # just keep instance as is w/ all attributes (i.e., no need to do anything)
    instance_dict = instance_obj.dict('endogenous')

  # convert instance (dictionary) to instance (array) to input into sklearn model
  instance_array = np.expand_dims(np.array(list(instance_dict.values())), axis=0)

  if prediction_class == 'positive':
    if args.classifier_class in global_vars.FAIR_MODELS and args.dataset_class != 'adult':
      return objs.classifier_obj.predict(instance_array)[0] == 1
    else:
      return objs.classifier_obj.predict_proba(instance_array)[0][1] > 0.5
  elif prediction_class == 'negative':
    if args.classifier_class in global_vars.FAIR_MODELS and args.dataset_class != 'adult':
      return objs.classifier_obj.predict(instance_array)[0] == -1
    else:
      return objs.classifier_obj.predict_proba(instance_array)[0][1] <= .50 - args.epsilon_boundary
  else:
    raise NotImplementedError


def isPointConstraintSatisfied(args, objs, factual_instance_obj, action_set, recourse_type):
  counter_instance_obj = computeCounterfactualInstance(
    args,
    objs,
    factual_instance_obj,
    action_set,
    recourse_type,
  )
  # assert counter instance has prediction = `positive`
  return isPredictionOfInstanceInClass(args, objs, counter_instance_obj, 'positive')


def getPrediction(args, objs, instance_obj):

  # get instance for trained model
  if args.classifier_class in global_vars.FAIR_MODELS:

    fair_nodes = getTrainableNodesForFairModel(args, objs)
    instance_dict = dict(zip(
      fair_nodes,
      [instance_obj.dict('endogenous_and_exogenous')[key] for key in fair_nodes]
    ))

  else:
    # just keep instance as is w/ all attributes (i.e., no need to do anything)
    instance_dict = instance_obj.dict('endogenous')

  # convert instance (dictionary) to instance (array) to input into sklearn model
  instance_array = np.expand_dims(np.array(list(instance_dict.values())), axis=0)
  prediction = objs.classifier_obj.predict(instance_array)[0]
  assert prediction in {0, 1}, f'Expected prediction in {0,1}; got {prediction}'
  return prediction


def getRecourseDistributionSample(args, objs, factual_instance_obj, action_set, recourse_type, num_samples):

  assert recourse_type in global_vars.ACCEPTABLE_DISTR_RECOURSE

  if not bool(action_set): # if action_set is empty, CFE = F
    return pd.DataFrame(dict(zip(
      objs.dataset_obj.getInputAttributeNames(),
      [num_samples * [factual_instance_obj.dict()[node]] for node in objs.dataset_obj.getInputAttributeNames()],
    )))

  samples_df = sampling._sampleInnerLoop(args, objs, factual_instance_obj, action_set, recourse_type, num_samples)

  return samples_df # return the entire data frame


def computeLowerConfidenceBound(args, objs, factual_instance_obj, action_set, recourse_type):
  if args.classifier_class in global_vars.FAIR_MODELS and args.dataset_class != 'adult':
    # raise NotImplementedError # cannot raise error, because runRecourseExperiment() call this to evaluate as a metric
    print('[WARNING] computing lower confidence bound with SVM model using predict_proba() may not work as intended.')
    return -1
  monte_carlo_samples_df = getRecourseDistributionSample(
    args,
    objs,
    factual_instance_obj,
    action_set,
    recourse_type,
    args.num_mc_samples,
  )

  if args.classifier_class in global_vars.FAIR_MODELS:
    fair_nodes = getTrainableNodesForFairModel(args, objs)
    # TODO (medpri): will not work for cw-fair-svm, because we
    # do not have the noise variables in this dataframe; add noise variables
    # (abducted and/or true) to this dataframe
    monte_carlo_samples_df = monte_carlo_samples_df[fair_nodes]
  monte_carlo_predictions = objs.classifier_obj.predict_proba(monte_carlo_samples_df)[:,1] # class 1 probabilities.

  expectation = np.mean(monte_carlo_predictions)
  # variance = np.sum(np.power(monte_carlo_predictions - expectation, 2)) / (len(monte_carlo_predictions) - 1)
  std = np.std(monte_carlo_predictions)

  # return expectation, variance

  # IMPORTANT... WE ARE CONSIDERING {0,1} LABELS AND FACTUAL SAMPLES MAY BE OF
  # EITHER CLASS. THEREFORE, THE CONSTRAINT IS SATISFIED WHEN SIGNIFICANTLY
  # > 0.5 OR < 0.5 FOR A FACTUAL SAMPLE WITH Y = 0 OR Y = 1, RESPECTIVELY.

  if getPrediction(args, objs, factual_instance_obj) == 0:
    return expectation - args.lambda_lcb * std # NOTE DIFFERNCE IN SIGN OF STD
  else: # factual_prediction == 1
    raise Exception(f'Should only be considering negatively predicted individuals...')
    # return expectation + args.lambda_lcb * np.sqrt(variance) # NOTE DIFFERNCE IN SIGN OF STD


def isDistrConstraintSatisfied(args, objs, factual_instance_obj, action_set, recourse_type):
  return computeLowerConfidenceBound(args, objs, factual_instance_obj, action_set, recourse_type) > 0.5


def getIndexOfFactualInstanceInDataFrame(factual_instance_obj, data_frame):
  # data_frame may include X and U, whereas factual_instance_obj.dict() only includes X
  assert set(factual_instance_obj.dict().keys()).issubset(set(data_frame.columns))

  matching_indicies = []
  for enumeration_idx, (factual_instance_idx, row) in enumerate(data_frame.iterrows()):
    if np.all([
      factual_instance_obj.dict()[key] == row[key]
      for key in factual_instance_obj.dict().keys()
    ]):
      matching_indicies.append(enumeration_idx)

  if len(matching_indicies) == 0:
    raise Exception(f'Was not able to find instance in data frame.')
  elif len(matching_indicies) > 1:
    raise Exception(f'Multiple matching instances are found in data frame: {matching_indicies}')
  else:
    return matching_indicies[0]
