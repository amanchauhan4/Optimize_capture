# use optimization to solve dynamics and optimal control

import numpy as np
import casadi as ca
import math as mt
import matplotlib.pyplot as plt
import os.path
n = 3  #number of drones
g = 9.81

#define the parameters
md = 1.0   #mass of each drone
m = 1.0 #mass of payload
tf_guess = 20.0 #guess optimal time
#kv = 0.0
N = 100  #Number of time intervals
kmax = 3.0 #maximum allowable jerk
#aerodynamic parameters
Cd = 2.0
rho=1.225
Sref = 1.0
#define reference values
#Used to scale model equations for better convergence
mu_ref = 1.00*100
t_ref = 1/mu_ref**0.5
u_ref = kmax*t_ref
l_ref=u_ref*t_ref**2
v_ref = l_ref/t_ref

#define internal forces as optimization variables
mu_array = []
program_zero = 1e-3

for i in range(n):
    for j in range(i+1,n+1):
        var_name = 'mu_'+str(i)+str(j)
        mu_array=ca.vertcat(mu_array,ca.SX.sym(var_name))
print(mu_array)


def mu_index(i,j):
    #function to convert drone index numbers to array index
    in1 = min(i,j)
    in2 = max(i,j)
    return (5*in1-in1**2)/2+in2-1

#define state and control variables
x = ca.SX.sym('x',6*(n+1))
u = ca.SX.sym('u',3*n)


#define Adjacency matrix
Ad=ca.SX.zeros(n+1,n+1)

for i in range(n+1):
   for j in range(n+1):
        if i!=j:
            Ad[i,j]=mu_array[mu_index(i,j)]

#Calculate the laplacian
L = ca.diag(Ad@ca.SX.ones((n+1,1)))-Ad

#calculate the system dynamics
A = ca.vertcat(ca.horzcat(ca.SX.zeros((3*n+3,3*n+3)),ca.SX.eye(3*n+3)),ca.horzcat(-ca.kron(L,ca.SX.eye(3)),ca.SX.zeros((3*n+3,3*n+3))))
#A = ca.vertcat(ca.horzcat(ca.SX.zeros((3*n+3,3*n+3)),ca.SX.eye(3*n+3)),ca.horzcat(-ca.kron(L,ca.SX.eye(3)),-ca.kron(Lv,ca.SX.eye(3))))

#print(A)
B = ca.SX.eye(6*n+6)
B = B[0:,3*n+5:]
xdot = A@x+B@ca.vertcat(-g/u_ref,u)
#system dynamics
k = ca.SX.sym('k',3*n)
s=ca.vertcat(x,u)
sdot = ca.vertcat(xdot,k)

#casADi function for state equations
f = ca.Function('f', [s,k,mu_array], [sdot], ['s','k','mu'], ['sdot'])


# solve the optimization problem

#define intial conditions
#drone positions
drone_pi=[]
drone_pi.append([-1.732,1.0,0.0])
drone_pi.append([1.732,1,0.0])
drone_pi.append([0.0,-2.0,0.0])
drone_pi = np.array(drone_pi)/l_ref
#drone velocity
drone_vi = []
drone_vi.append([0.0,0.0,0.0])
drone_vi.append([0.0,0.0,0.0])
drone_vi.append([0.0,0.0,0.0])
drone_vi = np.array(drone_vi)/v_ref

#define final conditions

drone_pf = []
drone_pf.append([-1.732/2.0,1/2.0,0.0])
drone_pf.append([1.732/2.0,1/2.0,0.0])
drone_pf.append([0,-1.0,0.0])
drone_pf = np.array(drone_pf)/l_ref

drone_vf = []
drone_vf.append([0.0,0.0,0.0])
drone_vf.append([0.0,0.0,0.0])
drone_vf.append([0.0,0.0,0.0])
drone_vf = np.array(drone_vf)/v_ref


#payload intial conditons
target_pi = []
target_pi = np.array([0.0,0.0,0.0])/l_ref #position
target_vi = np.array([0.0,0.0,-1.0])/v_ref #velocity



#initial conditions
x1 = np.array(drone_pi[0])
x2 = np.array(drone_pi[1])
x3 = np.array(drone_pi[2])
xo = np.array(target_pi)

v1 = np.array(drone_vi[0])
v2 = np.array(drone_vi[1])
v3 = np.array(drone_vi[2])
vo = np.array(target_vi)
#get initial state vector
x0 = np.concatenate((xo,x1,x2,x3,vo,v1,v2,v3))


lo = np.zeros((n+1,n+1))

for i in range(n+1):
    ri = x0[3*i:3*i+3]
    for j in range(n+1):
        rj = x0[3*j:3*j+3]  
        if i!=j:
            lo[i,j] = np.linalg.norm(ri-rj)
        else:
            lo[i,j] = 0
#initial conditions
#calculate final payload position
catch_point = np.zeros(3)
for i in range(n):
    catch_point=catch_point+np.array(drone_pf[i])
catch_point=catch_point/n

target_zf = -(lo[0,1]**2-np.linalg.norm(catch_point-drone_pf[0])**2)**0.5

target_pf = catch_point+np.array([0,0,target_zf])


u_start = np.zeros(9)
u_final = []
#the final control input must be consistent with the system model

for i in range(n):
    u_final.append(m*g/md/n/(drone_pf[i,2]-target_pf[2])*(drone_pf[i]-target_pf)/u_ref)
u_final=np.concatenate(u_final)
s_start = np.concatenate([x0,u_start])

xf = np.concatenate([target_pf,drone_pf[0],drone_pf[1],drone_pf[2],[0.0,0.0,0.0],drone_vf[0],drone_vf[1],drone_vf[2]])

s_end = np.concatenate([xf,u_final])

xg = []
k_guess = []
#linear initial guess
for i in range(N):
    xg.append(s_start+(s_end-s_start)*i/(N-1))

#u_guess=np.array(k_guess)


# Start with an empty NLP
w=[]
w0 = []

lbw = []
ubw = []
J = 0
g=[]
lbg = []
ubg = []




Xc = []
Ukc = []
mukc = []
#print(xo)
# For plotting x and u given w
x_plot = []
u_plot = []
mu_plot = []

#define upper and lower bounds
position_lb = -np.inf
position_ub = np.inf
velocity_lb = -np.inf
velocity_ub = np.inf
u_lb = -np.inf
u_ub = np.inf

x_lb = np.concatenate([position_lb*np.ones(3*(n+1)),velocity_lb*np.ones(3*(n+1)),u_lb*np.ones(9)])
x_ub = np.concatenate([position_ub*np.ones(3*(n+1)),velocity_ub*np.ones(3*(n+1)),u_ub*np.ones(9)])

k_lb = -1*np.ones(3*n)
k_ub = 1*np.ones(3*n)
#define final time
Xk = ca.SX.sym('tf')
Xc.append(Xk)
w.append(Xk)
lbw.append([0.0])
ubw.append([np.inf])
w0.append([tf_guess/t_ref])

#intialize constarint varaibles
mu_array_t = []
mu_array_t_all = []
for i in range(n+1):
    for j in range(i+1,n+1):
        var_name = 'mu_0_'+str(i)+str(j)
        muk=ca.SX.sym(var_name)
        mu_array_t=ca.vertcat(mu_array_t,muk)
        
        mukc.append(muk)
        
        w.append(muk)
        lbw.append([0.0])
        ubw.append([np.inf])
        w0.append([10.0/mu_ref])
mu_array_t_all.append(mu_array_t)
mu_plot.append(mu_array_t)


# initial conditions
Xk = ca.SX.sym('X_0',6*(n+1)+3*n)
Xc.append(Xk)
w.append(Xk)
lbw.append(s_start)
ubw.append(s_start)
w0.append(s_start)
x_plot.append(Xk)


Uk = ca.SX.sym('k_0',3*n)
Ukc.append(Uk)
w.append(Uk)
lbw.append(k_lb)
ubw.append(k_ub)
w0.append(0.5*(k_lb+k_ub))
u_plot.append(Uk)
#include constraints on mu

for i in range(n):
    ri = Xk[3*i:3*i+3]
    for j in range(i+1,n+1):
        rj = Xk[3*j:3*j+3]
        var_name = 'mu_0_'+str(i)+str(j)
        g.append(mu_array_t[mu_index(i,j)]*((ca.norm_2(ri-rj))-lo[i,j]))
        
        lbg.append([0.0])
        ubg.append([0.0])
        g.append((ca.norm_2(ri-rj))-lo[i,j])
       
        lbg.append([-np.inf])
        ubg.append([0.0])



for i in range(1,N-1):
    
    Xk = ca.SX.sym('X_'+str(i),6*(n+1)+3*n)
    Xc.append(Xk)
    w.append(Xk)
    lbw.append(x_lb)
    ubw.append(x_ub)
    w0.append(xg[i])
    x_plot.append(Xk)
    
    mu_array_t = [] 
    for i2 in range(n):
        for j2 in range(i2+1,n+1):
            var_name = 'mu_'+str(i)+'_'+str(i2)+str(j2)
            muk=ca.SX.sym(var_name)
            mu_array_t=ca.vertcat(mu_array_t,muk)
            mukc.append(muk)
            
            w.append(muk)
            lbw.append([0.0])
            ubw.append([np.inf])
            w0.append([1000.0/mu_ref])
    mu_plot.append(mu_array_t)
    mu_array_t_all.append(mu_array_t)

    
    # New NLP variable for the control
    Uk = ca.SX.sym('k_' + str(i),3*n)
    Ukc.append(Uk)
    w.append(Uk)
    lbw.append(k_lb)
    ubw.append(k_ub)
    w0.append(0.5*(k_lb+k_ub))
    u_plot.append(Uk)
    
    # Append collocation equations
    fj  = f(Xc[i],Ukc[i-1],mu_array_t_all[i-1])
    fjp1  = f(Xc[i+1],Ukc[i],mu_array_t_all[i])
    g.append(Xc[i+1]-Xc[i]-Xc[0]/(N-1)*0.5*(fj+fjp1))
    lbg.append(np.zeros(6*(n+1)+3*n))
    ubg.append(np.zeros(6*(n+1)+3*n))
    
    
    
    
    for i2 in range(n):
        ri = Xk[3*i2:3*i2+3]
        for j2 in range(i2+1,n+1):
            rj = Xk[3*j2:3*j2+3]
            var_name = 'mu_'+str(i)+'_'+str(i2)+str(j2)
         
            g.append(mu_array_t[mu_index(i2,j2)]*((ca.norm_2(ri-rj))-lo[i2,j2]))
            
            lbg.append([0.0])
            ubg.append([0.0])
            g.append((ca.norm_2(ri-rj))-lo[i2,j2])
            lbg.append([-np.inf])
            ubg.append([0.0])


i=N-1
Xk = ca.SX.sym('X_'+str(i),6*(n+1)+3*n)
Xc.append(Xk)
w.append(Xk)
lbw.append(s_end)
ubw.append(s_end)
w0.append(s_end)
x_plot.append(Xk)
#mu_plot.append(mu_array_t)
#u_plot.append(Uk)

mu_array_t = [] 
for i2 in range(n):
    for j2 in range(i2+1,n+1):
        var_name = 'mu_'+str(i)+'_'+str(i2)+str(j2)
        muk=ca.SX.sym(var_name)
        mu_array_t=ca.vertcat(mu_array_t,muk)
        mukc.append(muk)
        
        w.append(muk)
        lbw.append([0.0])
        ubw.append([np.inf])
        w0.append([1000.0/mu_ref])
mu_plot.append(mu_array_t)
mu_array_t_all.append(mu_array_t)


for i2 in range(n):
    ri = Xk[3*i2:3*i2+3]
    for j2 in range(i2+1,n+1):
        rj = Xk[3*j2:3*j2+3]
        var_name = 'mu_'+str(i)+'_'+str(i2)+str(j2)
        
        g.append(mu_array_t[mu_index(i2,j2)]*((ca.norm_2(ri-rj))-lo[i2,j2]))
        
        lbg.append([0.0])
        ubg.append([0.0])
        g.append((ca.norm_2(ri-rj))-lo[i2,j2])
      
        lbg.append([-np.inf])
        ubg.append([0.0])


# New NLP variable for the control
Uk = ca.SX.sym('k_' + str(i),3*n)
Ukc.append(Uk)
w.append(Uk)
lbw.append(k_lb)
ubw.append(k_ub)
w0.append(0.5*(k_lb+k_ub))
u_plot.append(Uk)

fj  = f(Xc[i],Ukc[i-1],mu_array_t_all[i-1])
fjp1  = f(Xc[i+1],Ukc[i],mu_array_t_all[i])
g.append(Xc[i+1]-Xc[i]-Xc[0]/(N-1)*0.5*(fj+fjp1))
lbg.append(np.zeros(6*(n+1)+3*n))
ubg.append(np.zeros(6*(n+1)+3*n))



# Concatenate vectors
w = ca.vertcat(*w)
g = ca.vertcat(*g)
x_plot = ca.horzcat(*x_plot)
u_plot = ca.horzcat(*u_plot)
mu_plot = ca.horzcat(*mu_plot)

w0 = np.concatenate(w0)
lbw = np.concatenate(lbw)
ubw = np.concatenate(ubw)
lbg = np.concatenate(lbg)
ubg = np.concatenate(ubg)


#define objective function

J = w[0]*t_ref

# Create an NLP solver
prob = {'x': w, 'f':J, 'g': g}
opts = {'ipopt':{'max_iter':5000}}
solver = ca.nlpsol('solver', 'ipopt', prob,opts)
# Function to get x and u trajectories from w

#trajectories = ca.Function('trajectories', [w], [x_plot, u_plot], ['s'], ['x', 'u'])
# Solve the NLP
#print(np.size(lbw))
sol = solver(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)
output = sol['x']
output = np.array(output)
t_array = np.concatenate([0.0+i*output[0]/(N-1) for i in range(N)])

# Function to get x and u trajectories from w
trajectories = ca.Function('trajectories', [w], [x_plot,u_plot,mu_plot], ['w'], ['x', 'u','mu'])
#trajectories = ca.Function('trajectories', [w], [x_plot,u_plot], ['w'], ['x', 'u'])
x_opt, u_opt, mu_opt = trajectories(sol['x'])
#x_opt, u_opt = trajectories(sol['x'])
x_opt = x_opt.full() # to numpy array
u_opt = u_opt.full() # to numpy array
mu_opt = mu_opt.full() # to numpy array
x_opt=x_opt
u_opt=u_opt
save_path='D:\MSAAE\Fall_2024\AIMS_research\model_try2'
filename = 'Output.dat'
fname=os.path.join(save_path,filename)
fid = open(fname,'w')
for i in range(len(t_array)):
    fid.write(f"{t_array[i]*t_ref}  {u_opt[0,i]*kmax} {u_opt[1,i]*kmax}   {u_opt[2,i]*kmax}  {x_opt[1,i]*l_ref}    {x_opt[2,i]*l_ref}    {x_opt[3,i]*l_ref}    {x_opt[4,i]*l_ref}    {x_opt[5,i]*l_ref}    {x_opt[6,i]*l_ref}   {x_opt[15,i]*v_ref} {x_opt[18,i]*v_ref} {x_opt[25,i]*u_ref} {x_opt[26,i]*u_ref} {x_opt[27,i]*u_ref}   \n")
fid.close()

#print(x_opt[0:,-1]-s_end)
plt.plot(t_array*t_ref,u_opt[2]*kmax)
plt.show()


plt.plot(t_array*t_ref,x_opt[5]*l_ref)
plt.show()

plt.plot(t_array*t_ref,x_opt[8]*l_ref)
plt.show()

plt.plot(t_array*t_ref,x_opt[11]*l_ref)
plt.show()

for i in range(6):
    plt.plot(t_array*t_ref,mu_opt[i]*mu_ref)
    plt.show()
