from third_party_code.pruning import * #Chen group Methods
import copy

def applyPruningRegroup(model,initialization,pruning_rate,train_loader,multithread=False):
    '''
    Apply chen regroup + IMP pruning:
        1.Prune the model using standard L1 pruning
        2.Extract an optimal mask from this pruning
        3.Regroup the weights using the compact sparse format described in: paper
        4.Remove pruning layer
        5.Reset weights to initial values
        6.Use the custom "regroup-friendly" pruning method.
    '''

    #Step1:
    pruning_model(model, pruning_rate, conv1=False)
    remain_weight = check_sparsity(model, conv1=False)
    #Step2:
    current_mask = extract_mask(model.state_dict())
    current_mask_copy = copy.deepcopy(current_mask)
    #Step3:
    for m in tqdm(current_mask_copy):
        mask = current_mask_copy[m]
        shape = mask.shape
        current_mask[m] = regroup(mask.view(mask.shape[0], -1)).view(*shape)
    #Step4:
    remove_prune(model, conv1=False)
    #Step5:
    model.load_state_dict(initialization)
    #Step6:
    prune_model_custom(model, current_mask)

    check_sparsity(model, conv1=False)

    return

def applyPruningRefill(model,initialization,pruning_rate,train_loader,multithread=False):
    '''
    Apply chen refill + IMP pruning:
        1.Prune the model using standard L1 pruning
        2.Extract an optimal mask from this pruning
        3.Remove pruning layer
        4.Reset weights to initial values
        5.Use the custom "refill-friendly" pruning method.
    '''
    #Step1:
    pruning_model(model, pruning_rate, conv1=False)
    remain_weight = check_sparsity(model, conv1=False)
    #Step2:
    current_mask = extract_mask(model.state_dict())
    for m in current_mask:
        print(current_mask[m].float().mean())
    #Step3:
    remove_prune(model, conv1=False)

    model.load_state_dict(initialization)
    prune_model_custom_fillback(model, current_mask, train_loader=train_loader)
    check_sparsity(model, conv1=False)

def applyDummyPruningRoutine(model,initialization,pruning_rate,train_loader):
    '''
    This routine do nothing, simply placeholder when we do not want to prune during training
    '''