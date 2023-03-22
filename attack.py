import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

from utils import normalize_cifar

def attack_pgd(model, x, y, eps, alpha, n_iters, dataset='cifar10' ,norm='Linf'):
    delta = torch.zeros_like(x).to(x.device)
    if norm == 'Linf':
        delta.uniform_(-eps, eps)
    else:
        raise ValueError
    delta = torch.clamp(delta, 0-x, 1-x)

    delta.requires_grad = True
    for _ in range(n_iters):
        if dataset == 'cifar10':
            output = model(normalize_cifar(x+delta))
        else:
            raise ValueError
        loss = F.cross_entropy(output, y)
        loss.backward()
        grad = delta.grad.detach()
        d = torch.clamp(delta + alpha * torch.sign(grad), min=-eps, max=eps)
        d = torch.clamp(d, 0 - x, 1 - x)
        delta.data = d
        delta.grad.zero_()
    
    return delta.detach()

def cw_attack_pgd(model, x, y, eps, alpha, n_iters,norm='Linf'):
    delta = torch.zeros_like(x).to(x.device)
    base = torch.ones_like(x).to(x.device)
    for sample in range(len(x)):
        base[sample] *= eps[sample]
    eps = base.clone().to(x.device)

    if norm == 'Linf':
        delta = (torch.rand_like(delta) - 0.5) * 2
        delta.to(x.device)
        delta = delta * eps  # remargin
    else:
        raise ValueError

    delta = torch.clamp(delta, 0-x, 1-x)
    delta.requires_grad = True
    for _ in range(n_iters):
        output = model(normalize_cifar(x+delta))
        loss = F.cross_entropy(output, y)
        loss.backward()
        grad = delta.grad.detach()
        d = torch.clamp(delta + alpha * torch.sign(grad), min=-eps, max=eps)
        d = torch.clamp(d, 0 - x, 1 - x)
        delta.data = d
        delta.grad.zero_()
    
    return delta.detach()

def pgd_loss(model, x, y, eps, beta, alpha, n_iters=10):
    delta = attack_pgd(model, x, y, eps, alpha, n_iters)
    robust_output = model(normalize_cifar(x + delta))
    criterion = nn.CrossEntropyLoss()
    return criterion(robust_output, y), robust_output.clone().detach()

def cw_pgd_loss(model, x, y, cw_eps, beta, alpha, n_iters=10):
    batch_eps = cw_eps[y]
    delta = cw_attack_pgd(model, x, y, batch_eps, alpha, n_iters)
    robust_output = model(normalize_cifar(x + delta))
    criterion = nn.CrossEntropyLoss()
    return criterion(robust_output, y), robust_output.clone().detach()

def trades_loss(model, x_natural, y, epsilon, cw_beta, step_size=0.003, perturb_steps=10):
    # define KL-loss
    criterion_kl = nn.KLDivLoss(size_average=False)
    model.eval()
    batch_size = len(x_natural)
    # generate adversarial example
    x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).to(x_natural.device).detach()
    for _ in range(perturb_steps):
        x_adv.requires_grad_()
        with torch.enable_grad():
            loss_kl = criterion_kl(F.log_softmax(model(normalize_cifar(x_adv)), dim=1),
                                    F.softmax(model(normalize_cifar(x_natural)), dim=1))
        grad = torch.autograd.grad(loss_kl, [x_adv])[0]
        x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
        x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
        x_adv = torch.clamp(x_adv, 0.0, 1.0)

    model.train()

    x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
    # calculate robust loss
    logits = model(normalize_cifar(x_natural))
    loss_natural = F.cross_entropy(logits, y, reduction='none') / batch_size

    cw_criterion = nn.KLDivLoss(reduction='none')
    robust_out = model(normalize_cifar(x_adv))
    loss_robust = (1.0 / batch_size) * cw_criterion(F.log_softmax(robust_out, dim=1),
                                                    F.softmax(model(normalize_cifar(x_natural)), dim=1))
    batch_beta = cw_beta[y]
    assert len(batch_beta) == len(loss_robust)                                                
    loss_robust = loss_robust.sum(1)
    #print(batch_beta.shape, loss_natural.shape, loss_robust.shape)
    loss = ((1-batch_beta) * loss_natural + batch_beta * loss_robust).sum()

    return loss, robust_out.clone().detach()

def cw_trades_loss(model, x_natural, y, cw_eps, cw_beta, step_size, perturb_steps):
    batch_beta = cw_beta[y]
    batch_eps = cw_eps[y]
    base = torch.ones_like(x_natural).to(x_natural.device)
    for sample in range(len(x_natural)):
        base[sample] *= batch_eps[sample]
    batch_eps = base.clone().to(x_natural.device)

    # define KL-loss
    criterion_kl = nn.KLDivLoss(size_average=False)
    model.eval()
    batch_size = len(x_natural)
    # generate adversarial example
    x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).to(x_natural.device).detach()
    for _ in range(perturb_steps):
        x_adv.requires_grad_()
        with torch.enable_grad():
            loss_kl = criterion_kl(F.log_softmax(model(normalize_cifar(x_adv)), dim=1),
                                    F.softmax(model(normalize_cifar(x_natural)), dim=1))
        grad = torch.autograd.grad(loss_kl, [x_adv])[0]
        x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
        x_adv = torch.min(torch.max(x_adv, x_natural - batch_eps), x_natural + batch_eps)
        x_adv = torch.clamp(x_adv, 0.0, 1.0)

    model.train()

    x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
    # calculate robust loss
    logits = model(normalize_cifar(x_natural))
    loss_natural = F.cross_entropy(logits, y, reduction='none') / batch_size
    cw_criterion = nn.KLDivLoss(reduction='none')
    robust_out = model(normalize_cifar(x_adv))
    loss_robust = (1.0 / batch_size) * cw_criterion(F.log_softmax(robust_out, dim=1),
                                                    F.softmax(model(normalize_cifar(x_natural)), dim=1))

    assert len(batch_beta) == len(loss_robust)                                                
    loss_robust = loss_robust.sum(1)
    #print(batch_beta.shape, loss_natural.shape, loss_robust.shape)
    loss = ((1-batch_beta) * loss_natural + batch_beta * loss_robust).sum()

    return loss, robust_out.clone().detach()

def fat_loss(model, x, y, eps, tau, alpha, n_iters=10):
    K = n_iters
    control = (torch.ones(len(x)) * tau).to(x.device)
    delta = torch.zeros_like(x).to(x.device)
    delta.uniform_(-eps, eps)
    delta = torch.clamp(delta, 0-x, 1-x)
    
    
    delta.requires_grad = True
    output = model(normalize_cifar(x+delta))
    pred = output.max(1)[1]

    while K > 0:
        iter_index = []
        for idx in range(len(y)):
            if pred[idx] != y[idx] and control[idx] > 0:
                control[idx] -= 1
                iter_index.append(idx)
            elif pred[idx] == y[idx] and control[idx] > 0:
                iter_index.append(idx)
            
        if len(iter_index) != 0:

            loss = F.cross_entropy(output, y)
            loss.backward()
            grad = delta.grad.detach()
            g = grad[iter_index]
            d = torch.clamp(delta[iter_index] + alpha * torch.sign(g), min=-eps, max=eps)
            d = torch.clamp(d, 0 - x[iter_index], 1 - x[iter_index])
            
            delta = delta.detach()
            delta[iter_index] = d.detach()

            delta.requires_grad_()
            output = model(normalize_cifar(x+delta))
            pred = output.max(1)[1]

        else:
            break
        K -= 1
    
    robust_output = model(normalize_cifar(x + delta))
    criterion = nn.CrossEntropyLoss()
    return criterion(robust_output, y), robust_output.clone().detach()

def cw_fat_loss(model, x, y, cw_eps, tau, alpha, n_iters=10):
    batch_eps = cw_eps[y]
    delta = torch.zeros_like(x).to(x.device)
    base = torch.ones_like(x).to(x.device)
    for sample in range(len(x)):
        base[sample] *= batch_eps[sample]
    batch_eps = base.clone().to(x.device)

    delta = (torch.rand_like(delta) - 0.5) * 2
    delta.to(x.device)
    delta = delta * batch_eps
    delta = torch.clamp(delta, 0-x, 1-x)

    K = n_iters
    control = (torch.ones(len(x)) * tau).to(x.device)

    delta.requires_grad = True
    output = model(normalize_cifar(x+delta))
    pred = output.max(1)[1]

    while K > 0:
        iter_index = []
        for idx in range(len(y)):
            if pred[idx] != y[idx] and control[idx] > 0:
                control[idx] -= 1
                iter_index.append(idx)
            elif pred[idx] == y[idx] and control[idx] > 0:
                iter_index.append(idx)
            
        if len(iter_index) != 0:

            loss = F.cross_entropy(output, y)
            loss.backward()
            grad = delta.grad.detach()
            g = grad[iter_index]
            d = torch.clamp(delta[iter_index] + alpha * torch.sign(g),
                 min=-batch_eps[iter_index], max=batch_eps[iter_index])
            d = torch.clamp(d, 0 - x[iter_index], 1 - x[iter_index])
            
            delta = delta.detach()
            delta[iter_index] = d.detach()

            delta.requires_grad_()
            output = model(normalize_cifar(x+delta))
            pred = output.max(1)[1]

        else:
            break
        K -= 1
    
    robust_output = model(normalize_cifar(x + delta))
    criterion = nn.CrossEntropyLoss()
    return criterion(robust_output, y), robust_output.clone().detach()
