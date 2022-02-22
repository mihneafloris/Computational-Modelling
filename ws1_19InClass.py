#================================================================= 
#
# AE2220-II: Computational Modelling 
# Code for work session 1
#
#=================================================================
# This code provides a base for computing the linearised 
# perturbation potential around a slender 2D body 
# symmetric about the x axis. Some key lines in the code include:
#
# lines 23-35:  Input parameters 
# lines 56-60:  Definition of the body geometry
# lines 96-103: Implementation of finite-difference scheme.
# 
#=================================================================

import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


#=========================================================
# Input parameters
#=========================================================
Mach       = 2.0;                # Mach number
Uinf       = 1.0;                # Freestream velocity
targArea   = 0.10;               # Body target area

x1         = -1.0;               # Forward boundary position
x2         = 2.0;                # Rear boundary position
y2         = 1.0;                # Upper boundary position
imax       = 40;                 # Number of mesh points in i
jmax       = 20;                 # Number of mesh points in j

plots      = 1;                  # Make plots if not zero
stride     = 1;                  # Point skip rate for suface plot
maxl       = 50;                 # maximum grid lines on plots


#=========================================================
# Derived parameters
#=========================================================
beta      = math.sqrt(Mach*Mach-1);  # sqrt(M^2-1)
dx        = (x2-x1)/(imax-1);        # Mesh spacing in x
dy        = (y2)/(jmax-1);           # Mesh spacing in y


#=========================================================
# Define the lower boundary (perturbation) geometry 
# and compute its derivative theta=dy/dx
#=========================================================
xlower = np.linspace(x1, x2, imax);
ylower = np.zeros(imax);
theta  = np.zeros(imax);

# Body surface definition
for i in range(0, imax):
  if xlower[i] > 0.0 and xlower[i]<1.0:
     xl = xlower[i];
     ylower[i] = 0.30*(1-xl)*(xl); 
     
     
# Body surface Derivative 
# (dy/dx = theta assumed zero at x=x1 and x=x2)
for i in range(1, imax-1):
 theta[i] = (ylower[i+1]-ylower[i-1])/(2*dx);



#=========================================================
# Load the mesh coordinates and
# compute theta, the dy/dx of the body geometry
#=========================================================
x      = np.zeros((imax,jmax));          # Mesh x coordinates
y      = np.zeros((imax,jmax));          # Mesh y coordinates

# Mesh coordinates
for j in range(0, jmax):
  x[:,j] = np.linspace(x1, x2, imax)

for i in range(0, imax):
  y[i,:] = np.linspace(0, y2, jmax)



#=========================================================
# March explicitly in x, solving for the unknown 
# Riemann invariant, R
#=========================================================
R = np.zeros((imax,jmax));  # Note R(0,:)=0 for i=0;

#**************************************
# Uncomment the "for" and "R" lines 
# below then add code where requested
#**************************************
for n in range(0, imax-1):   # March from x=0 to x=2
  
   # Apply boundary condition at y=0
  R[n+1,0] = 2* theta[n+1]
   
   # Update interior values using a first-order accurate upwind scheme
  for j in range(1, jmax):
    R[n+1,j] = R[n,j] - 1/beta *dx * (R[n,j]-R[n,j-1])/dy



#=========================================================
# Compute velocities, cp, area, and wave drag
#=========================================================
u=np.zeros((imax,jmax));
v=np.zeros((imax,jmax));
cp=np.zeros((imax,jmax));

for i in range(imax):
  for j in range(jmax):
     v[i,j]  =  R[i,j]/2.
     u[i,j]  = -v[i,j]/beta;
     cp[i,j] = -2*u[i,j]/Uinf;

area    = 0.
drag    = 0.
for i in range(imax-1):
   xmid   = (xlower[i]+xlower[i+1])/2;
   ymid   = (ylower[i]+ylower[i+1])/2;
   pmid   = (cp[i,0]+cp[i+1,0])/2;
   tmid   = (theta[i+1]+theta[i])/2;
   dpdx   = (cp[i+1,0]-cp[i,0])/(xlower[i+1]-xlower[i]);
   area   += 2.*dx*ymid;
   drag   += 2.*dx*pmid*tmid;


q=dx/(dy*beta);
print ("------------------------------------------")
print ("Summary:")
print ("------------------------------------------")
print ("Mach, Beta     = ",Mach,",",beta)
print ("imax x jmax    = ",imax,"x",jmax)
print ("q=dx/(dy*beta) = ",dx/(dy*beta))
print ("area/target    = ",area/targArea)
print ("body drag/opt  = ",drag/(12*targArea*targArea/beta))
print ("body drag      = ",drag)
print ("cp(2,1)        = ",cp[imax-1,jmax-1])
print ("------------------------------------------")

#------------------------------------------------------
#  Plot results
#------------------------------------------------------
if plots != 0:

 fig = plt.figure(figsize=(18,10))

 ax1 = plt.subplot2grid((2,4),(0,0),colspan=2,rowspan=2,projection='3d')
 ax1.set_xlabel('x')
 ax1.set_ylabel('y')
 ax1.set_zlabel('R')
 ax1.plot_surface(x, y, R, shade=True, rstride=stride,cstride=stride,
 cmap=plt.cm.CMRmap, linewidth=0, antialiased=True);

 ax3 = plt.subplot2grid((2,4), (0,2),colspan=2)
 ax3.set_title(r"$v(x,y) \;and\; \theta(x)$")
 a = ax3.contourf(x, y, v, cmap=plt.cm.jet)
 if (imax<maxl) & (jmax<maxl):
   ax3.plot(x, y, '-k', x.transpose(), y.transpose(), '-k')
 ax3.plot(xlower,0.2*theta,'--',linewidth=2.0,color='blue')
 fig.colorbar(a, ax=ax3)

 ax4 = plt.subplot2grid((2,4), (1,2),colspan=2)
 ax4.set_title(r"$c_p(x,y) \;and\; ylower(x)$")
 a = ax4.contourf(x, y, cp, cmap=plt.cm.jet)
 if (imax<maxl) & (jmax<maxl):
  ax4.plot(x, y, '-k', x.transpose(), y.transpose(), '-k')
 ax4.plot(xlower,ylower,linewidth=2.0,color='blue')
 fig.colorbar(a, ax=ax4)

 ax1.view_init(30, -120)
 plt.savefig('super_' + str(imax) + '_' + str(q) + '.png',dpi=250)
 plt.show()

#------------------------------------------------------
#  All done
#------------------------------------------------------
print ("done")
