function sg_ctld
% soccer game by continuous time learning dynamics
close all; 

global sta_space act_space bct_space P R D gamma rho0 epsilon eta
global AwinState BwinState DrawState

data = load('sg_dp.mat'); 
P = data.P;
R = data.R*100; 
D = data.D; 
gamma = data.d;          % discounted factor 
disp(['load ' num2str(size(P)) ' probability transition matrix']); 
disp(['load ' num2str(size(R)) ' reward vector']); 
disp(['load ' num2str(size(D)) ' terminal vector']); 
sta_space = data.sta_space; 
act_space = data.act_space; 
bct_space = data.bct_space; 
AwinState = data.AwinState; 
BwinState = data.BwinState; 
DrawState = data.DrawState; 

rho0 = ones(sta_space+3,1) / (sta_space);         % s0 distribution 
rho0([AwinState BwinState DrawState]) = 0;      % terminal state has no initial distribution 

epsilon = 0.001; 
% epsilon = 0.05;
% epsilon = 0.01;
% epsilon = 0.02;
% epsilon = 0.1;

eta = 1;
% eta = 10;
% eta = 0.5;
% eta = 3;

y0 = zeros(sta_space*(act_space+bct_space), 1);

% tspan = 0:0.1:10; 
tspan = 0:0.01:0.2; 
opts = odeset('OutputFcn', @odeplot); 
[t,y] = ode45(@sg_mgfp, tspan, y0, opts); 

figure; plot(t,y);

y1 = y(:,1:sta_space*act_space);
y1 = reshape(y1, length(t), sta_space, act_space);
x1 = bsxfun(@rdivide, exp(y1 /epsilon), sum(exp(y1/epsilon), 3));

piA = squeeze(x1(end,:,:)); 
disp(['error ' num2str(norm(piA(:)-data.piA(:)))]); 

save(sprintf('sg_ctld_epsilon(%g)_eta(%g).mat', epsilon, eta)); 

end


function dy = sg_mgfp(t, y)
% soccor game Markov game fictitious-play dynamics
global sta_space act_space bct_space P R D gamma rho0 epsilon eta

y1 = reshape(y(1:sta_space*act_space), sta_space, act_space);           % stateAB * action 
y2 = reshape(y(sta_space*act_space+1:end), sta_space, bct_space); 

pi1 = bsxfun(@rdivide, exp(y1/epsilon), ...
    sum(exp(y1/epsilon), 2));               % state * action
pi2 = bsxfun(@rdivide, exp(y2/epsilon), ...
    sum(exp(y2/epsilon), 2));               % state * action

Ppi = bsxfun(@times, P, ...         % s * a1 * a2 * sn
    bsxfun(@times, pi1, ...         % s * a1
    permute(pi2, [1 3 2])));        % s * 1 * a2
Ppi = squeeze(sum(Ppi, [2 3]));             % s * sn
rho_pi = (eye(sta_space+3) - gamma*Ppi'*eye(sta_space, sta_space+3) ...
    - gamma*diag([zeros(1,sta_space) 1 1 1])) \ rho0;        % (s+3) * 1

Vpi1 = (eye(sta_space) - gamma*Ppi*eye(sta_space+3, sta_space)) \ Ppi * R;          % s*1
Vpi2 = -Vpi1; 

Ppi2 = bsxfun(@times, P, ...        % s * a1 * a2 * sn
    permute(pi2, [1 3 2]));         % s * 1 * a2 
Ppi2 = squeeze(sum(Ppi2, 3));       % s * a1 * sn: state transition matrix given player 2 policy
q1 = R + gamma * eye(sta_space+3, sta_space) * Vpi1;      % sn * 1
Qpi1 = bsxfun(@times, Ppi2, ...     % s * a1 * sn
    permute(q1, [3 2 1]));          % 1 * 1 * sn
Qpi1 = squeeze(sum(Qpi1, 3));       % s * a1
Api1 = Qpi1 - Vpi1; 

v1 = bsxfun(@times, rho_pi(1:sta_space), ...     % s * 1
    Api1);                          % s * a1
v1 = reshape(v1, [], 1); 

Ppi1 = bsxfun(@times, P, ...        % s * a1 * a2 * sn
    pi1);                           % s * a1
Ppi1 = squeeze(sum(Ppi1, 2));       % s * a2 * sn: state transition matrix given player 1 policy
q2 = - R + gamma * eye(sta_space+3, sta_space) * Vpi2;      % sn * 1
Qpi2 = bsxfun(@times, Ppi1, ...     % s * a1 * sn
    permute(q2, [3 2 1]));          % 1 * 1 * sn
Qpi2 = squeeze(sum(Qpi2, 3));       % s * a1
Api2 = Qpi2 - Vpi2; 

v2 = bsxfun(@times, rho_pi(1:sta_space), ...     % s * 1
    Api2);                          % s * a2
v2 = reshape(v2, [], 1); 

dy = eta * ([v1;v2] - y); 

end