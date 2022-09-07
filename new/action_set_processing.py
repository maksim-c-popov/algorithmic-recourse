import itertools
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

import shap
import json

import new.global_vars as global_vars
import new.utils as utils
import new.grad_descent as grad_descent
from new.classes.instance import Instance


def getValidDiscretizedActionSets(args, objs):

  possible_actions_per_node = []

  # IMPORTANT: you lose ordering of columns when using setdiff! This should not
  # matter in this part of the code, but may elsewhere. For alternative, see:
  # https://stackoverflow.com/questions/46261671/use-numpy-setdiff1d-keeping-the-order
  intervenable_nodes = np.setdiff1d(
    objs.dataset_obj.getInputAttributeNames('kurz'),
    list(
      np.unique(
        list(args.non_intervenable_nodes) +
        list(args.sensitive_attribute_nodes)
      )
    )
  )

  for attr_name_kurz in intervenable_nodes:

    attr_obj = objs.dataset_obj.attributes_kurz[attr_name_kurz]

    if attr_obj.attr_type in {'numeric-real', 'numeric-int', 'binary'}:

      if attr_obj.attr_type == 'numeric-real':
        number_decimals = 5
      elif attr_obj.attr_type in {'numeric-int', 'binary'}:
        number_decimals = 0

      # bad code pseudonym; don't access internal object attribute
      tmp_min = objs.dataset_obj.data_frame_kurz.describe()[attr_name_kurz]['min']
      tmp_max = objs.dataset_obj.data_frame_kurz.describe()[attr_name_kurz]['max']
      tmp_mean = objs.dataset_obj.data_frame_kurz.describe()[attr_name_kurz]['mean']
      tmp = list(
        np.around(
          np.linspace(
            tmp_mean - 2 * (tmp_mean - tmp_min),
            tmp_mean + 2 * (tmp_max - tmp_mean),
            args.grid_search_bins
          ),
          number_decimals,
        )
      )
      tmp.append('n/a')
      tmp = list(dict.fromkeys(tmp))
      # remove repeats from list; this may happen, say for numeric-int, where we
      # can have upper-lower < args.grid_search_bins, then rounding to 0 will result
      # in some repeated values
      possible_actions_per_node.append(tmp)

    else:

      # select N unique values from the range of fixed categories/ordinals
      tmp = list(
        np.unique(
          np.round(
            np.linspace(
              attr_obj.lower_bound,
              attr_obj.upper_bound,
              args.grid_search_bins
            )
          )
        )
      )
      tmp.append('n/a')
      possible_actions_per_node.append(tmp)

  all_action_tuples = list(itertools.product(
    *possible_actions_per_node
  ))

  all_action_tuples = [
    elem1 for elem1 in all_action_tuples
    if len([
      elem2 for elem2 in elem1 if elem2 != 'n/a'
    ]) <= args.max_intervention_cardinality
  ]

  all_action_sets = [
    dict(zip(intervenable_nodes, elem))
    for elem in all_action_tuples
  ]

  # Go through, and for any action_set that has a value = 'n/a', remove ONLY
  # THAT key, value pair, NOT THE ENTIRE action_set.
  valid_action_sets = []
  for action_set in all_action_sets:
    valid_action_sets.append({k:v for k,v in action_set.items() if v != 'n/a'})

  return valid_action_sets


def measureActionSetCost(args, objs, factual_instance_obj_or_ts, action_set, processing_type = 'raw', range_normalized = True):
  # TODO (medpri): this function is called using both brute-force and grad-descent
  # approach (where the former/latter uses Instance/dict)
  if isinstance(factual_instance_obj_or_ts, Instance):
    factual_instance = factual_instance_obj_or_ts.dict()
  else:
    factual_instance = factual_instance_obj_or_ts

  X_all = utils.processDataFrameOrInstance(args, objs, utils.getOriginalDataFrame(objs, args.num_train_samples), processing_type)
  ranges = dict(zip(
    X_all.columns,
    [np.max(X_all[col]) - np.min(X_all[col]) for col in X_all.columns],
  ))
  # TODO (medpri): WHICH TO USE??? FOR REPRODUCABILITY USE ABOVE!
  # (above uses ranges in train set, below over all the dataset)
  # ranges = objs.dataset_obj.getVariableRanges()

  if not range_normalized:
    ranges = {key: 1 for key in ranges.keys()}

  if \
    np.all([isinstance(elem, (int, float)) for elem in factual_instance.values()]) and \
    np.all([isinstance(elem, (int, float)) for elem in action_set.values()]):
    deltas = [
      (action_set[key] - factual_instance[key]) / ranges[key]
      for key in action_set.keys()
    ]
    return np.linalg.norm(deltas, args.norm_type)
  elif \
    np.all([isinstance(elem, torch.Tensor) for elem in factual_instance.values()]) and \
    np.all([isinstance(elem, torch.Tensor) for elem in action_set.values()]):
    deltas = torch.stack([
      (action_set[key] - factual_instance[key]) / ranges[key]
      for key in action_set.keys()
    ])
    return torch.norm(deltas, p=args.norm_type)
  else:
    raise Exception(f'Mismatching or unsupport datatypes.')


def getValidInterventionSets(args, objs):
  # IMPORTANT: you lose ordering of columns when using setdiff! This should not
  # matter in this part of the code, but may elsewhere. For alternative, see:
  # https://stackoverflow.com/questions/46261671/use-numpy-setdiff1d-keeping-the-order
  intervenable_nodes = np.setdiff1d(
    objs.dataset_obj.getInputAttributeNames('kurz'),
    list(
      np.unique(
        list(args.non_intervenable_nodes) +
        list(args.sensitive_attribute_nodes)
      )
    )
  )

  all_intervention_tuples = utils.powerset(intervenable_nodes)
  all_intervention_tuples = [
    elem for elem in all_intervention_tuples
    if len(elem) <= args.max_intervention_cardinality
    and elem is not tuple() # no interventions (i.e., empty tuple) could never result in recourse --> ignore
  ]

  return all_intervention_tuples


def computeOptimalActionSet(args, objs, factual_instance_obj, save_path, recourse_type):

  # assert factual instance has prediction = `negative`
  # assert isPredictionOfInstanceInClass(args, objs, factual_instance_obj, 'negative')
  # TODO (fair): bring back the code above; can't do so with twin_factual_instances.
  if not utils.isPredictionOfInstanceInClass(args, objs, factual_instance_obj, 'negative'):
    return {} # return empty action set for those twin_factual_instances that are not negatively predicted

  if recourse_type in global_vars.ACCEPTABLE_POINT_RECOURSE:
    constraint_handle = utils.isPointConstraintSatisfied
  elif recourse_type in global_vars.ACCEPTABLE_DISTR_RECOURSE:
    constraint_handle = utils.isDistrConstraintSatisfied
  else:
    raise Exception(f'{recourse_type} not recognized.')

  if args.optimization_approach == 'brute_force':

    valid_action_sets = getValidDiscretizedActionSets(args, objs)
    print(f'\n\t[INFO] Computing optimal `{recourse_type}`: grid searching over {len(valid_action_sets)} action sets...')

    min_cost = np.infty
    min_cost_action_set = {}
    for action_set in tqdm(valid_action_sets):
      if constraint_handle(args, objs, factual_instance_obj, action_set, recourse_type):
        cost_of_action_set = measureActionSetCost(args, objs, factual_instance_obj, action_set)
        if cost_of_action_set < min_cost:
          min_cost = cost_of_action_set
          min_cost_action_set = action_set

    print(f'\t done (optimal action set: {str(min_cost_action_set)}).')

  elif args.optimization_approach == 'grad_descent':

    valid_intervention_sets = getValidInterventionSets(args, objs)
    print(f'\n\t[INFO] Computing optimal `{recourse_type}`: grad descent over {len(valid_intervention_sets)} intervention sets (max card: {args.max_intervention_cardinality})...')

    min_cost = np.infty
    min_cost_action_set = {}
    
    result_action_sets = []
    result_costs = []

    for idx, intervention_set in enumerate(valid_intervention_sets):
      # print(f'\n\t[INFO] intervention set #{idx+1}/{len(valid_intervention_sets)}: {str(intervention_set)}')
      # plotOptimizationLandscape(args, objs, factual_instance_obj, save_path, intervention_set, recourse_type)
      action_set, recourse_satisfied, cost_of_action_set = grad_descent.performGDOptimization(args, objs, factual_instance_obj, save_path, intervention_set, recourse_type)
      if constraint_handle(args, objs, factual_instance_obj, action_set, recourse_type):
        assert recourse_satisfied # a bit redundant, but just in case, we check
                                  # that the MC samples from constraint_handle()
                                  # on the line above and the MC samples from
                                  # performGradDescentOptimization() both agree
                                  # that recourse has been satisfied
        assert np.isclose( # won't be exact becuase the former is float32 tensor
          cost_of_action_set,
          measureActionSetCost(args, objs, factual_instance_obj, action_set),
          atol = 1e-2,
        )

        result_action_sets.append(action_set)
        result_costs.append(cost_of_action_set)

        if cost_of_action_set < min_cost:
          min_cost = cost_of_action_set
          min_cost_action_set = action_set


    print('====================')
    print(f'Done (optimal intervention set: {str(min_cost_action_set)}).')
    print(f'Results for all intervention sets:')

    result_sets_with_cost = list(zip(result_action_sets, result_costs))
    result_sets_with_cost.sort(key=lambda x: x[1])

    [print("Cost: " + str(round(res[1], 6)) + "; Action set: " + json.dumps(res[0])) for res in result_sets_with_cost]
    print('====================')
    
    X_all = utils.getOriginalDataFrame(objs, args.num_train_samples)

    explainer = shap.Explainer(objs.classifier_obj.predict, X_all)

    factial_x = pd.DataFrame([factual_instance_obj.dict()])

    shap_values = explainer(factial_x)

    intervenable_nodes = np.setdiff1d(
    objs.dataset_obj.getInputAttributeNames('kurz'),
    list(
      np.unique(
        list(args.non_intervenable_nodes) +
        list(args.sensitive_attribute_nodes)
      )
    ))

    relevant_indexes = [shap_values[0].feature_names.index(el) for el in intervenable_nodes]

    relevant_shap_values = list(np.array(list(zip(shap_values[0].feature_names, abs(shap_values[0].values))))[relevant_indexes])
    relevant_shap_values.sort(key=lambda x: x[1], reverse=True)

    relevant_shap_feature_names = [x[0] for x in relevant_shap_values]

    shap_intervention_sets = []
    shap_set_lenght = min(len(relevant_shap_values), args.max_shap_intervention_cardinality)
    for i in range(1, shap_set_lenght + 1):
      shap_intervention_sets.append(set(relevant_shap_feature_names[:i]))
    
    print('relevant shap values:')
    print(relevant_shap_values)
    print('shap intervention_sets:')
    print(shap_intervention_sets)
    print('====================')

    result_shap_action_sets = []
    result_shap_costs = []

    for _, intervention_set in enumerate(shap_intervention_sets):
      action_set, recourse_satisfied, cost_of_action_set = grad_descent.performGDOptimization(args, objs, factual_instance_obj, save_path, intervention_set, recourse_type)
      if constraint_handle(args, objs, factual_instance_obj, action_set, recourse_type):
        assert recourse_satisfied # a bit redundant, but just in case, we check
        assert np.isclose( # won't be exact becuase the former is float32 tensor
          cost_of_action_set,
          measureActionSetCost(args, objs, factual_instance_obj, action_set),
          atol = 1e-2,
        )

        result_shap_action_sets.append(action_set)
        result_shap_costs.append(cost_of_action_set)
    
    print('====================')
    print(f'Results for all SHAP intervention sets:')

    result_sets_with_cost = list(zip(result_shap_action_sets, result_shap_costs))
    result_sets_with_cost.sort(key=lambda x: x[1])

    [print("Cost: " + str(round(res[1], 6)) + "; Action set: " + json.dumps(res[0])) for res in result_sets_with_cost]
    print('====================')

  else:
    raise Exception(f'{args.optimization_approach} not recognized.')

  return min_cost_action_set