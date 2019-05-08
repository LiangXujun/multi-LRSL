#%%
import pandas as pd
import numpy as np
from numpy import sqrt, zeros, eye, argsort, diag, ones, trace
from numpy.linalg import solve, norm
from scipy.spatial.distance import pdist, squareform

#%%
def soft_threshold(Z, a):
    G = np.where(Z-a > 0, Z-a, 0) - np.where(-Z-a > 0, -Z-a, 0)
    return G


def get_knn(data_mat, k, similarity = 'cosine'):
    n = data_mat.shape[0]
    
    S = 1 - squareform(pdist(data_mat + np.finfo(np.float).eps, similarity)) - eye(n)
    
    ind_knn = argsort(-S, 1)
    Sknn = zeros((n, n))
    for j in range(n):
        ind = ind_knn[j,0:k]
        Sknn[j,ind] = S[j,ind]
        Sknn[ind,j] = S[ind,j]
    
    return Sknn
    
def mlrsl(train_data, params, m = 0, max_iter = 100, tol = 1e-4):
    Xtrain = train_data['Xtrain']
    Ytrain = train_data['Ytrain']
    Ls = train_data['Ls']
    R0 = train_data['R0']
    
    mu = params['mu']
    lam = params['lam']
    beta = params['beta']
    alpha = params['alpha']
    gam = params['gam']
    
    if m == 0:
        m = len(Xtrain)
    n, l = Ytrain.shape
    d = [Xtrain[i].shape[1] for i in range(m)]
    G = [zeros((d[i], l)) for i in range(m)]
    R = np.random.uniform(size = (l, l))
    theta = ones(m)/m
    t1 = 1
    t2 = 1
    check_step = 1
    total_loss_old = 1
    
    for ii in range(0, max_iter):
        Gold = [G[i].copy() for i in range(m)]
        
        Q = Ytrain
        L = zeros((n, n))
        for i in range(m):
            Q = Q + mu*Xtrain[i]@G[i]
            L = L + theta[i]**gam*Ls[i]
        P = (1 + m*mu)*eye(n) + L
        F = solve(P, Q)
        
        Rn = 0
        Rd = 0
        for i in range(m):
            GG = G[i].T@G[i]
            GG_pos = (abs(GG) + GG)/2
            GG_neg = (abs(GG) - GG)/2
            Rn += GG_pos
            GG_diag = diag(diag(GG))
            Rd += GG_neg + GG_diag@ones((l, l)) + ones((l, l))@GG_diag - GG_diag
        Rn = lam/2*Rn + 2*alpha*R0
        Rd = lam/2*Rd + 2*alpha*R
        R[Rd!=0] = R[Rd!=0]*sqrt(Rn[Rd!=0]/Rd[Rd!=0])
        LR = diag(R.sum(1)) - R
        
        for i in range(m):
            Gpk = G[i] + (t1 - 1)/t2*(G[i] - Gold[i])
            A = mu*Xtrain[i].T@Xtrain[i] - mu**2*Xtrain[i].T@solve(P.T, Xtrain[i])
            dfGpk = A@Gpk - mu*Xtrain[i].T@solve(P, Ytrain) + lam*Gpk@LR
            for j in range(m):
                if i == j:
                    continue
                else:
                    dfGpk = dfGpk - mu**2*Xtrain[i].T@solve(P.T, Xtrain[j])@Gold[j]
            Lf = sqrt(2*norm(A, 2)**2 + 2*norm(lam*LR, 2)**2)
            Zk = Gpk - 1/Lf*dfGpk
            G[i] = soft_threshold(Zk, beta/Lf)
        
        t1, t2 = t2, (1 + sqrt(4*t2**2 + 1))/2
        
        for i in range(m):
            theta[i] = (1/trace(F.T@Ls[i]@F))**(1/(gam - 1))
        theta = theta/theta.sum()
        
        for i in range(m):
            L = L + theta[i]**gam*Ls[i]
        
        predict_loss = 1/2*norm(F - Ytrain, 'fro')**2 + alpha*norm(R - R0, 'fro')**2
        correlation = 1/2*trace(F.T@L@F)
        sparse = 0
        for i in range(m):
            predict_loss += mu/2*norm(Xtrain[i]@G[i] - F, 'fro')**2
            correlation += lam/2*trace(LR@G[i].T@G[i])
            sparse += beta*sum(sum(abs(G[i])))
        total_loss = predict_loss + correlation + sparse
        
        loss_perc = abs((total_loss_old - total_loss)/total_loss_old)
        if ii%check_step == 0:
            print("ii = %i, loss = %.4f, loss perc = %.4f" %(ii, total_loss, loss_perc))
    
        if loss_perc < tol:
            break
        else:
            total_loss_old = total_loss
    return G, F, theta, ii 

#%% 
chem_mat = pd.read_csv(r"drug_fp_mat.txt", sep = "\t", header = 0, index_col = 0)
dom_mat = pd.read_csv("./drug_domain_mat.txt", sep = "\t", header = 0, index_col = 0)
go_mat = pd.read_csv("./drug_gobp_mat.txt", sep = "\t", header = 0, index_col = 0)
expr_mat = pd.read_csv("./drug_lincs_gene_mat.txt", sep = "\t", header = 0, index_col = 0)
side_mat = pd.read_csv("./drug_pt_mat.txt", sep = "\t", header = 0, index_col = 0)

chemmat_new = pd.read_csv('./drug_fp_mat_case_study.txt', sep = "\t", header = 0, index_col = 0);
dommat_new = pd.read_csv('./drug_dom_mat_case_study.txt', sep = "\t", header = 0, index_col = 0);
gomat_new = pd.read_csv('./drug_go_mat_case_study.txt', sep = "\t", header = 0, index_col = 0);
exprmat_new = pd.read_csv('./drug_expr_mat_case_study.txt', sep = "\t", header = 0, index_col = 0);

#%%
X1 = chem_mat.to_numpy()
X2 = dom_mat.to_numpy()
X3 = go_mat.to_numpy()
X4 = expr_mat.to_numpy()
Y = side_mat.to_numpy()

X1_test = chemmat_new.to_numpy()
X2_test = dommat_new.to_numpy()
X3_test = gomat_new.to_numpy()
X4_test = exprmat_new.to_numpy()

#%%
nfold = 5
mu = 0.1
beta = 0.01
lam = 1
alpha = 1
gam = 2

#%%
Xtrain = [X1, X2, X3, X4]
Xtest = [X1_test, X2_test, X3_test, X4_test]
Ytrain = Y
n, l = Ytrain.shape
ntest = X1_test.shape[0]
m = len(Xtrain)

R0 = get_knn(Ytrain.T, np.int(l*0.01))

Ls = []
for i in range(m):
    Sknn = get_knn(Xtrain[i], np.int(n*0.01), 'cosine')
    Ls.append(diag(Sknn.sum(1)) - Sknn)


Xtest = [np.hstack((X, ones((ntest, 1)))) for X in Xtest] #add bias
Xtrain = [np.hstack((X, ones((n, 1)))) for X in Xtrain]


train_data = {'Xtrain':Xtrain, 'Ytrain':Ytrain, 'Ls':Ls, 'R0':R0}
params = {'mu':mu, 'lam':lam, 'beta':beta, 'alpha':alpha, 'gam':gam}

G, _, theta, _ = mlrsl(train_data, params)

Yscore = zeros((ntest, l))
for i in range(m):
    Yscore += theta[i]*Xtest[i]@G[i]
