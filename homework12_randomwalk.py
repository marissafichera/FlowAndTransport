## By: Jesus D. Gomez
## Date: Spring 2009
## Last update: 04-15-2016
# This function estimates the journey of a set of particles following a
# random walk
function[points] = RandomWalkModel_2016()
clear
close all
clc
## Data for random walk
dt = 1; #time step in seconds
numTimeStep = 200; #number of time steps
diffCoef = 0; #diffussion coefficient
dispCoef_L = 1; # longitudinal dispersivity (m) # THIS IS SILLY NOTATION.  Should
be alpha
dispCoef_T = 1; # transverse dispersivity (m)
ts=0:dt:numTimeStep*dt;
## Initial distribution of particles
# A strip of points from y=-100 to y = 100 at at x = 0
# dely = 100;
#xini = 0;
#y = -dely:dely;
#numPoints = (2*dely+1); #number of particles
# All points on the origin
numPoints = 1000; #number of particles
xini = 0;
yini = 0;
x = xini.*ones(1,numPoints);
y = yini.*ones(1,numPoints);
points0 = [x' y']; #Initial distribution of particles
###################################################################################
#########
# Plot initial distribution
figure(1)
set(gcf,'defaultlinemarkersize',6); set(gcf,'defaultlinelinewidth',2);
set(gcf,'defaultaxesfontsize',12);
plot(x,y,'*k')
title('\bf Initial distribution of particles'), ylabel('\bf y [m]'), xlabel('\bf x
[m]')
set(gca,'FontWeight','bold')
###################################################################################
##########
#Set up basic bookkeeping for particles. The third dimension stores all
#the particles at different time steps
points = zeros(length(x),2,numTimeStep+1);
points(:,:,1) = points0; # Store initial distribution
for i = 2:numTimeStep+1;

    #Develop velocity information
    [u,v] = flowField(points0(:,1),points0(:,2));
    velocity = [u v];
    [alphaL,alphaT] = Disp(points0(:,1),points0(:,2),dispCoef_L,dispCoef_T);

    #Displacements
    dx = u.*dt;
    dy = v.*dt;
    ds = [dx dy];
    #Speed (magnitude of the velocity vector)
    speed = sqrt(u.^2 + v.^2);

    #Find principal directions
    longDir = velocity./[speed speed];
    tranDir = [longDir(:,2) -longDir(:,1)];
    #Compute longitudinal and transverse variances
    varLong = 2.*(alphaL.*speed + diffCoef).*dt;
    varTrans = 2.*(alphaT.*speed + diffCoef).*dt;
    #Reinitialize random number generator
    randn('state',sum(100*clock));
    #Create normally distributed random vector
    Zt = normrnd(0,sqrt(varTrans),numPoints,1);
    Zl = normrnd(0,sqrt(varLong),numPoints,1);

    #Advect and then introduce random noise
    points0(:,1) = points0(:,1) + dx + Zt.*(dy./speed) + Zl.*(dx./speed); # he
doesn't use longDir and tranDir; computes the "angle" here
    points0(:,2) = points0(:,2) + dy - Zt.*(dx./speed) + Zl.*(dy./speed);

    points(:,1,i)=points(:,1,i-1) + dx + Zt.*(dy./speed) + Zl.*(dx./speed);
    points(:,2,i) =points(:,2,i-1) + dy - Zt.*(dx./speed) + Zl.*(dy./speed);

    #Save new points in matrix for later analysis
#    points(:,:,i) = points0;

    # calculate statistics
    ms(1,:)=mean(points(:,1,:)); # mean x coordinate
    ms(2,:)=mean(points(:,2,:)); # mean y coordinate
    squeeze(ms);
    fit(1,:)=polyfit(ts,ms(1,:),1);
    fit(2,:)=polyfit(ts,ms(2,:),1);

    vs(1,:)=var(points(:,1,:)); # variance of x coordinate
    vs(2,:)=var(points(:,2,:)); # variance of y coordinate
    squeeze(vs);
    vfit(1,:)=polyfit(ts,vs(1,:),1);
    vfit(2,:)=polyfit(ts,vs(2,:),1);

    # final position variance
    fv(1)=vs(1,end);
    fv(2)=vs(2,end);

end
## Figures
figure(2)
#set(gcf,'defaultlinemarkersize',6); set(gcf,'defaultlinelinewidth',2);
set(gcf,'defaultaxesfontsize',12);
set(gcf,'defaultlinelinewidth',2); set(gcf,'defaultaxesfontsize',12);
plot(points(:,1,1),points(:,2,1),'*k',points(:,1,100),points(:,2,100),'or',...
    points(:,1,200),points(:,2,200),'ob'),...
    legend('t = 0s' ,'t = 100s','t = 200s')
title('\bf Isotropic Dispersion'), ylabel('\bf y [m]'), xlabel('\bf x [m]')
set(gca,'FontWeight','bold')
figure(3)
set(gcf,'defaultlinemarkersize',6); set(gcf,'defaultlinelinewidth',2);
set(gcf,'defaultaxesfontsize',12);
plot(points(:,1,numTimeStep),points(:,2,numTimeStep),'ok'),legend(strcat('t = 
',num2str(numTimeStep),'s'))
title('\bf Final distribution'), ylabel('\bf y [m]'), xlabel('\bf x [m]')
set(gca,'FontWeight','bold')
figure(4)
set(gcf,'defaultlinemarkersize',6); set(gcf,'defaultaxesfontsize',12);
plot(ts,ms(1,:),'og');
hold on;
plot(ts,ms(2,:),'om');
plot(ts,fit(1,1)*ts+fit(1,2),'k-')
plot(ts,fit(2,1)*ts+fit(2,2),'k--')
title('\bf Mean Particle Positions'), xlabel('\bf Time [s]'), ylabel('\bf Mean
Coordinate Position [m]')
legend('Mean x coordinate','Mean y coordinate',strcat('x slope = 
',num2str(fit(1,1)),' m/s'),strcat('y slope = ',num2str(fit(2,1)),' m/s'));
figure(5)
set(gcf,'defaultlinemarkersize',6); set(gcf,'defaultaxesfontsize',12);
plot(ts,vs(1,:),'og');
hold on;
plot(ts,vs(2,:),'om');
plot(ts,vfit(1,1)*ts+vfit(1,2),'k-')
plot(ts,vfit(2,1)*ts+vfit(2,2),'k--')
title('\bf Variance of Particle Positions'), xlabel('\bf Time [s]'), ylabel('\bf
Variance of Coordinate Positions [m]')
legend('Variance of x coordinates','Variance of y coordinates',strcat('x slope = 
',num2str(vfit(1,1)),' m/s'),strcat('y slope = ',num2str(vfit(2,1)),' m/s'));
disp('Done, my friend!!!')
disp('Final x variance:');disp(fv(1))
disp('Expected x:');disp(2*dispCoef_L*numTimeStep*dt); # NEED TO FIX THIS. SHOULD
BE 2*alpha*u*final time
disp('Expected y:');disp(2*dispCoef_T*numTimeStep*dt); # NEED TO FIX THIS. SHOULD
BE 2*alpha*u*final time
end