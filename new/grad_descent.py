import torch
import time
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

import new.global_vars as global_vars
import new.utils as utils
import new.sampling as sampling
import new.action_set_processing as action_set_processing
from new.classes.memoize import Memoize


@Memoize
def getTorchClassifier(args, objs):

  if isinstance(objs.classifier_obj, LogisticRegression):

    fixed_model_w = objs.classifier_obj.coef_
    fixed_model_b = objs.classifier_obj.intercept_
    fixed_model = lambda x: torch.sigmoid(
      (
        torch.nn.functional.linear(
          x,
          torch.from_numpy(fixed_model_w).float(),
        ) + float(fixed_model_b)
      )
    )

  elif isinstance(objs.classifier_obj, MLPClassifier):

    data_dim = len(objs.dataset_obj.getInputAttributeNames())
    fixed_model_width = 10 # TODO make more dynamic later and move to separate function
    assert objs.classifier_obj.hidden_layer_sizes == (fixed_model_width, fixed_model_width)
    fixed_model = torch.nn.Sequential(
      torch.nn.Linear(data_dim, fixed_model_width),
      torch.nn.ReLU(),
      torch.nn.Linear(fixed_model_width, fixed_model_width),
      torch.nn.ReLU(),
      torch.nn.Linear(fixed_model_width, 1),
      torch.nn.Sigmoid()
    )
    fixed_model[0].weight = torch.nn.Parameter(torch.tensor(objs.classifier_obj.coefs_[0].astype('float32')).t(), requires_grad=False)
    fixed_model[2].weight = torch.nn.Parameter(torch.tensor(objs.classifier_obj.coefs_[1].astype('float32')).t(), requires_grad=False)
    fixed_model[4].weight = torch.nn.Parameter(torch.tensor(objs.classifier_obj.coefs_[2].astype('float32')).t(), requires_grad=False)
    fixed_model[0].bias = torch.nn.Parameter(torch.tensor(objs.classifier_obj.intercepts_[0].astype('float32')), requires_grad=False)
    fixed_model[2].bias = torch.nn.Parameter(torch.tensor(objs.classifier_obj.intercepts_[1].astype('float32')), requires_grad=False)
    fixed_model[4].bias = torch.nn.Parameter(torch.tensor(objs.classifier_obj.intercepts_[2].astype('float32')), requires_grad=False)

  else:

    raise Exception(f'Converting {str(objs.classifier_obj.__class__)} to torch not supported.')

  X_all = utils.getOriginalDataFrame(objs, args.num_train_samples)
  assert np.all(
    np.isclose(
      objs.classifier_obj.predict_proba(X_all[:25])[:,1],
      fixed_model(torch.tensor(X_all[:25].to_numpy(), dtype=torch.float32)).flatten(),
      atol = 1e-3,
    )
  ), 'Torch classifier is not equivalent to the sklearn model.'

  return fixed_model


def performGDOptimization(args, objs, factual_instance_obj, save_path, intervention_set, recourse_type):

  def saveLossCurve(save_path, intervention_set, best_action_set_epoch, all_logs):
    fig, axes = plt.subplots(2 + len(intervention_set), 1, sharex=True)

    axes[0].plot(all_logs['epochs'], all_logs['loss_cost'], 'g--', label='loss costs')
    axes[0].plot(all_logs['epochs'], all_logs['loss_constraint'], 'r:', label='loss constraints')
    axes[0].plot(best_action_set_epoch, all_logs['loss_cost'][best_action_set_epoch-1], 'b*')
    axes[0].plot(best_action_set_epoch, all_logs['loss_constraint'][best_action_set_epoch-1], 'b*')
    axes[0].text(best_action_set_epoch, all_logs['loss_cost'][best_action_set_epoch-1], f"{all_logs['loss_cost'][best_action_set_epoch-1]:.3f}", fontsize='xx-small')
    axes[0].text(best_action_set_epoch, all_logs['loss_constraint'][best_action_set_epoch-1], f"{all_logs['loss_constraint'][best_action_set_epoch-1]:.3f}", fontsize='xx-small')
    axes[0].set_ylabel('loss', fontsize='xx-small')
    axes[0].set_title('Loss curve', fontsize='xx-small')
    axes[0].grid()
    axes[0].set_ylim(
      min(-1, axes[0].get_ylim()[0]),
      max(+1, axes[0].get_ylim()[1]),
    )
    axes[0].legend(fontsize='xx-small')

    axes[1].plot(range(1, len(all_logs['loss_total']) + 1), all_logs['lambda_opt'], 'y-', label='lambda_opt')
    axes[1].set_ylabel('lambda', fontsize='xx-small')
    axes[1].grid()
    axes[1].legend(fontsize='xx-small')

    for idx, node in enumerate(intervention_set):
      # print intervention values
      tmp = [elem[node] for elem in all_logs['action_set']]
      axes[idx+2].plot(all_logs['epochs'], tmp, 'b-', label='lambda_opt')
      axes[idx+2].set_ylabel(node, fontsize='xx-small')
      if idx == len(intervention_set) - 1:
        axes[idx+2].set_xlabel('epochs', fontsize='xx-small')
      axes[idx+2].grid()
      axes[idx+2].legend(fontsize='xx-small')

    plt.savefig(f'{save_path}/{str(intervention_set)}.pdf')
    plt.close()

  # IMPORTANT: if you process factual_instance_obj here, then action_set_ts and
  #            factual_instance_ts will also be normalized down-stream. Then
  #            at the end of this method, simply deprocess action_set_ts. One
  #            thing to note is the computation of distance may not be [0,1]
  #            in the processed settings (TODO (lowpri))
  if recourse_type in {'m0_true', 'm2_true'}:
    tmp_processing_type = 'raw'
  elif recourse_type in {'m1_alin', 'm1_akrr'}:
    tmp_processing_type = global_vars.PROCESSING_SKLEARN
  elif recourse_type in {'m1_gaus', 'm2_gaus'}:
    tmp_processing_type = global_vars.PROCESSING_GAUS
  elif recourse_type in {'m1_cvae', 'm2_cvae', 'm2_cvae_ps'}:
    tmp_processing_type = global_vars.PROCESSING_CVAE
  factual_instance_obj = utils.processDataFrameOrInstance(args, objs, factual_instance_obj, tmp_processing_type)

  # IMPORTANT: action_set_ts includes trainable params, but factual_instance_ts does not.
  factual_instance_ts = {k: torch.tensor(v, dtype=torch.float32) for k, v in factual_instance_obj.items()}


  def initializeNonSaturatedActionSet(args, objs, factual_instance_obj, intervention_set, recourse_type):
    # default action_set
    action_set = dict(zip(
      intervention_set,
      [
        factual_instance_obj.dict()[node]
        for node in intervention_set
      ]
    ))
    noise_multiplier = 0
    while noise_multiplier < 10:
      # create an action set from the factual instance, and possibly some noise
      action_set = {k : v + noise_multiplier * np.random.randn() for k,v in action_set.items()}
      # sample values
      if recourse_type in global_vars.ACCEPTABLE_POINT_RECOURSE:
        samples_df = sampling._sampleInnerLoop(args, objs, factual_instance_obj, action_set, recourse_type, 1)
      elif recourse_type in global_vars.ACCEPTABLE_DISTR_RECOURSE:
        samples_df = sampling._sampleInnerLoop(args, objs, factual_instance_obj, action_set, recourse_type, args.num_mc_samples)
      # return action set if average predictive probability of samples >= eps (non-saturated region of classifier)
      predict_proba_list = objs.classifier_obj.predict_proba(samples_df)[:,1]
      if np.mean(predict_proba_list) >= 5e-2 and np.mean(predict_proba_list) - 0.5: # don't want to start on the other side
        return action_set
      noise_multiplier += 0.1
    return action_set


  action_set = initializeNonSaturatedActionSet(args, objs, factual_instance_obj, intervention_set, recourse_type)
  action_set_ts = {k : torch.tensor(v, requires_grad = True, dtype=torch.float32) for k,v in action_set.items()}

  # TODO (lowpri): make input args
  min_valid_cost = 1e6  # some large number
  no_decrease_in_min_valid_cost = 0
  early_stopping_K = 10
  # DO NOT USE .copy() on the dict, the same value objects (i.e., the same trainable tensor will be used!)
  best_action_set = {k : v.item() for k,v in action_set_ts.items()}
  best_action_set_epoch = 1
  recourse_satisfied = False

  capped_loss = False
  num_epochs = args.grad_descent_epochs
  lambda_opt = 1 # initial value
  lambda_opt_update_every = 25
  lambda_opt_learning_rate = 0.5
  action_set_learning_rate = 0.1
  print_log_every = lambda_opt_update_every
  optimizer = torch.optim.Adam(params = list(action_set_ts.values()), lr = action_set_learning_rate)

  #all_logs = {}
  #all_logs['epochs'] = []
  #all_logs['loss_total'] = []
  #all_logs['loss_cost'] = []
  #all_logs['lambda_opt'] = []
  #all_logs['loss_constraint'] = []
  #all_logs['action_set'] = []

  start_time = time.time()
  if args.debug_flag:
    print(f'\t\t[INFO] initial action set: {str({k : np.around(v.item(), 4) for k,v in action_set_ts.items()})}') # TODO (lowpri): use pretty print

  # https://stackoverflow.com/a/52017595/2759976
  iterator = tqdm(range(1, num_epochs + 1))
  #iterator = range(1, num_epochs + 1)
  for epoch in iterator:

    # ========================================================================
    # CONSTRUCT COMPUTATION GRAPH
    # ========================================================================

    samples_ts = sampling._sampleInnerLoopTensor(args, objs, factual_instance_obj, factual_instance_ts, action_set_ts, recourse_type)

    # ========================================================================
    # COMPUTE LOSS
    # ========================================================================

    loss_cost = action_set_processing.measureActionSetCost(args, objs, factual_instance_ts, action_set_ts, tmp_processing_type)

    # get classifier
    h = getTorchClassifier(args, objs)
    pred_labels = h(samples_ts)

    # compute LCB
    if torch.isnan(torch.std(pred_labels)) or torch.std(pred_labels) < 1e-10:
      # When all predictions are the same (likely because all sampled points are
      # the same, likely because we are outside of the manifold OR, e.g., when we
      # intervene on all nodes and the initial epoch returns same samples), then
      # torch.std() will be 0 and therefore there is no gradient to pass back; in
      # turn this results in torch.std() giving nan and ruining the training!
      #     tmp = torch.ones((10,1), requires_grad=True)
      #     torch.std(tmp).backward()
      #     print(tmp.grad)
      # https://github.com/pytorch/pytorch/issues/4320
      # SOLUTION: remove torch.std() when this term is small to prevent passing nans.
      value_lcb = torch.mean(pred_labels)
    else:
      value_lcb = torch.mean(pred_labels) - args.lambda_lcb * torch.std(pred_labels)

    loss_constraint = (0.5 - value_lcb)
    if capped_loss:
      loss_constraint = torch.nn.functional.relu(loss_constraint)

    # for fixed lambda, optimize theta (grad descent)
    loss_total = loss_cost + lambda_opt * loss_constraint

    # ========================================================================
    # EARLY STOPPING
    # ========================================================================

    # check if constraint is satisfied
    if value_lcb.detach() > 0.5:
      # check if cost decreased from previous best
      if loss_cost.detach() < min_valid_cost:
        min_valid_cost = loss_cost.item()
        # DO NOT USE .copy() on the dict, the same value objects (i.e., the same trainable tensor will be used!)
        best_action_set = {k : v.item() for k,v in action_set_ts.items()}
        best_action_set_epoch = epoch
        recourse_satisfied = True
      else:
        no_decrease_in_min_valid_cost += 1

    # stop if past K valid thetas did not improve upon best previous cost
    if no_decrease_in_min_valid_cost > early_stopping_K:
      #saveLossCurve(save_path, intervention_set, best_action_set_epoch, all_logs)
      # https://stackoverflow.com/a/52017595/2759976
      iterator.close()
      break

    # ========================================================================
    # OPTIMIZE
    # ========================================================================

    # once every few epochs, optimize theta (grad ascent) manually (w/o pytorch)
    if epoch % lambda_opt_update_every == 0:
      lambda_opt = lambda_opt + lambda_opt_learning_rate * loss_constraint.detach()

    optimizer.zero_grad()
    loss_total.backward()
    optimizer.step()

    # ========================================================================
    # LOGS / IMAGES
    # ========================================================================

    if args.debug_flag and epoch % print_log_every == 0:
      print(
        f'\t\t[INFO] epoch #{epoch:03}: ' \
        f'optimal action: {str({k : np.around(v.item(), 4) for k,v in action_set_ts.items()})}    ' \
        f'loss_total: {loss_total.item():02.6f}    ' \
        f'loss_cost: {loss_cost.item():02.6f}    ' \
        f'lambda_opt: {lambda_opt:02.6f}    ' \
        f'loss_constraint: {loss_constraint.item():02.6f}    ' \
        f'value_lcb: {value_lcb.item():02.6f}    ' \
      )
    #all_logs['epochs'].append(epoch)
    #all_logs['loss_total'].append(loss_total.item())
    #all_logs['loss_cost'].append(loss_cost.item())
    #all_logs['lambda_opt'].append(lambda_opt)
    #all_logs['loss_constraint'].append(loss_constraint.item())
    #all_logs['action_set'].append({k : v.item() for k,v in action_set_ts.items()})

    #if epoch % 100 == 0:
      #saveLossCurve(save_path, intervention_set, best_action_set_epoch, all_logs)

  end_time = time.time()
  if args.debug_flag:
    print(f'\t\t[INFO] Done (total run-time: {end_time - start_time}).\n\n')

  # Convert action_set_ts to non-tensor action_set when passing back to rest of code.
  # best_action_set may or may not be result of early stopping, but it will
  # either be the initial value (which was zero-cost, at the factual instance),
  # or it will be the best_action_set seen so far (smallest cost and valid const)
  # whether or not it triggered K times to initiate early stopping.
  action_set = {k : v for k,v in best_action_set.items()}
  action_set = utils.deprocessDataFrameOrInstance(args, objs, action_set, tmp_processing_type)
  return action_set, recourse_satisfied, min_valid_cost
