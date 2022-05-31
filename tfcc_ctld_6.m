function tfcc_ctld_6
% three-firm cournot competition by continuous-time learning dynamics

close all 

global cap_space sta_space act_space MCs P Rs gamma rho0 epsilon eta

cap_space = 0:20:100;               % firm capacity, [0,100]
cap_size = length(cap_space); 
sta_space = cap_size^3;     % state space 
act_space = 3;                      % +10, -10, 0
MCs = [40,35,42];                   % marginal cost

P = zeros(sta_space, ...
    act_space, act_space, act_space, ...
    sta_space);           % state transition matrix, s*a1*a2*a3->sn
Rs = zeros(3, sta_space);                 % reward vector at next step state
prob_succ = 0.8; prob_fail = 0.1; 
for sta = 1:sta_space
    [x1,x2,x3] = ind2sub([cap_size cap_size cap_size], sta); 
    xs = [x1,x2,x3]; 
    xs = 20*(xs-1); 
    X = sum(xs); 
    price = 400 - 2*X; 
    for i = 1:3
        Rs(i,sta) = xs(i)*price - MCs(i)*xs(i); 
    end
    for Act = 1:act_space^3
        [a1,a2,a3] = ind2sub([act_space act_space act_space], Act); 
        acts = [a1,a2,a3]; 
        for noisy_Act = 1:act_space^3
            [a1,a2,a3] = ind2sub([act_space act_space act_space], noisy_Act); 
            noisy_acts = [a1,a2,a3]; 
            next_xs = xs;
            prob = 1; 
            for i = 1:3
                switch noisy_acts(i)
                    case 1
                        next_xs(i) = min(xs(i)+20,cap_space(end)); 
                    case 2
                        next_xs(i) = max(xs(i)-20,cap_space(1)); 
                    case 3
                        next_xs(i) = xs(i); 
                end
                if noisy_acts(i) == acts(i)
                    prob = prob * prob_succ; 
                else
                    prob = prob * prob_fail; 
                end
            end
            next_sta = (next_xs+20) / 20; 
            next_sta = sub2ind([cap_size cap_size cap_size], ...
                next_sta(1), next_sta(2), next_sta(3)); 
            P(sta, acts(1), acts(2), acts(3), next_sta) = ...
                P(sta, acts(1), acts(2), acts(3), next_sta) + prob; 
        end
    end
end

Rs = Rs / 10000; 
rho0 = ones(sta_space,1) / (sta_space);         % s0 distribution 
gamma = 0.9; 

% epsilon = 0.1; 
% epsilon = 0.01; 
% epsilon = 0.001;  
epsilon = 0.0001; 

eta = 1;

y0 = zeros(sta_space*act_space*3, 1);

% tspan = 0:0.1:10; 
tspan = 0:0.01:0.1; 
opts = odeset('OutputFcn', @odeplot); 
[t,y] = ode45(@tfcc_ctld_dyn, tspan, y0, opts); 

figure; plot(t,y);


save(sprintf('tfcc_ctld_size(%g)_epsilon(%g)_eta(%g).mat', cap_size, epsilon, eta)); 

end


function dy = tfcc_ctld_dyn(t, y)
% three firm cournot competition brea dynamics 
global cap_space sta_space act_space MCs P Rs gamma rho0 epsilon eta

ys = reshape(y, [], sta_space, act_space);    % 3*s*a
pis = bsxfun(@rdivide, exp(ys/epsilon), ...
    sum(exp(ys/epsilon), 3));           % 3*s*a

Ppi = P;        % s,a1,a2,a3->sn
for i = 1:3
    Ppi = bsxfun(@times, Ppi, squeeze(pis(i,:,:))); 
    Ppi = squeeze(sum(Ppi,2)); 
end
rho_pi = (eye(sta_space) - gamma*Ppi'*eye(sta_space)) \ rho0;        % s * 1

Vpis = zeros(3,sta_space);      % 3*s
for i = 1:3
    Vpis(i,:) = ((eye(sta_space) - gamma*Ppi*eye(sta_space)) \ Ppi * Rs(i,:)')'; 
end

Qpis = zeros(3,sta_space,act_space);    % 3*s*a
for i = 1:3
    Ppii = P;       % state transition without i-th policy
    for j = 1:3
        if j ~= i
            perm_index = [1,(3:j+1),2]; 
            Ppii = bsxfun(@times, Ppii, permute(squeeze(pis(j,:,:)), perm_index)); 
        end
    end
    for j = 1:3
        if j ~= i
            Ppii = sum(Ppii, j+1); 
        end
    end
    Ppii = squeeze(Ppii);       % s*a->sn
    qi = Rs(i,:)' + gamma*Vpis(i,:)';   % sn*1
    Qpis(i,:,:) = squeeze(sum(bsxfun(@times, Ppii, permute(qi, [2 3 1])),3)); 
end

Apis = bsxfun(@minus, Qpis, Vpis);      % 3*s*a
vs = bsxfun(@times, permute(rho_pi, [2 1 3]), Apis);    % 3*s*a

dy = eta * (reshape(vs, [], 1) - y); 

end

