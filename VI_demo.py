import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns

sigma = 10
k = 2
n = 10

def gen_data(sigma, k, n):
    #获取u_k
    u_k = np.random.normal(0, sigma, k)
    x = []
    c = []
    for i in range(n):
        ci = random.choice(range(k))
        c.append(ci)
        u = u_k[ci]
        x.append(np.random.normal(u, 1))
    return x, u_k, c

x, u_k, c = gen_data(sigma,k,n)

print("x:"+str(x))
print("u_k:"+str(u_k))
print("c:"+str(c))

sns.distplot(x, hist=False,color='y')
sns.distplot(x,color='y')
plt.show()

print('**'*100)

def solve(x, k, sigma, epoch=40):
    """
    x: 输入数据
    k: 超参数k，c_i的维度，在业务CASE中等于用户数
    sigma: 超参数，需要人工调整
    """
    n = len(x)
    phis = np.random.random([n, k])
    mk = np.random.random([k])
    sk = np.random.random([k])
    for _ in range(epoch):
        for i in range(n):
            phi_i_k = []
            for _k in range(k):
                #根据公式(6)更新参数phi_ik
                phi_i_k.append(np.exp(mk[_k]*x[i] - (sk[_k]**2 + mk[_k]**2)/2))
            sum_phi = sum(phi_i_k)
            phi_i_k = [phi/sum_phi for phi in phi_i_k]
            phis[i] = phi_i_k
        den = np.sum(phis, axis=0) + 1/(sigma**2)
        #根据公式(10)更新m_k
        mk = np.matmul(x, phis)/den
        #根据公式(11)更新s_k
        sk = np.sqrt(1/den)
    return mk, sk, phis

mk, sk, phis = solve(x,k,sigma)
n = 10 # number of sample to be drawn
samples = []
for i in range(n): # iteratively draw samples
    Z = np.random.choice([0,1]) # latent variable
    samples.append(np.random.normal(mk[Z], sk[Z], 1))

sns.distplot(x, hist=False,color='y')
sns.distplot(x,color='y')
sns.distplot(samples, hist=False,color='b')
sns.distplot(samples,color='b')
plt.show()
print("mk:"+str(mk))
print("sk:"+str(sk))
print("phis:"+str(phis))
pred = []
for p in phis:
    pred.append(np.argmax(p))
print(pred)
