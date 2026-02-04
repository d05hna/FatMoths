%% Load Data 
inf = h5info("teth_data.h5"); 
%disp(inf)

tmp = h5read("teth_data.h5",'/Hi_fx');
Hi_fx = complex(tmp.r,tmp.i);
tmp = h5read("teth_data.h5",'/Hf_fx');
Hf_fx = complex(tmp.r,tmp.i);

tmp = h5read("teth_data.h5",'/Hi_yaw');
Hi_yaw = complex(tmp.r,tmp.i);
tmp = h5read("teth_data.h5",'/Hf_yaw');
Hf_yaw = complex(tmp.r,tmp.i);

tmp = h5read("teth_data.h5",'/Hi_roll');
Hi_roll = complex(tmp.r,tmp.i);
tmp = h5read("teth_data.h5",'/Hf_roll');
Hf_roll = complex(tmp.r,tmp.i);

freqqs =  [0.2000,0.3000,0.5000,0.7000,1.100,1.300,1.700,1.900,2.300,2.900,3.700,4.300,5.300,6.100,7.900,8.900,11.30,13.70];
%% fit one moth for yaw 
m1 = Hi_yaw(1:15,1);
H = idfrd(m1,freqqs(1:15),1/10000);
sys = tfest(H,3,0);
mod_resp = squeeze(freqresp(sys,freqqs(1:15)));
figure; 
subplot(2,1,1)
scatter(freqqs(1:15),abs(m1),'b','filled');
hold on 
line(freqqs(1:15),abs(mod_resp))
title('3rd Order 1 Moth Yaw')
set(gca,'XScale','log')
set(gca,'Yscale','log')
hold off
subplot(2,1,2)
scatter(freqqs(1:15),unwrap(angle(m1)),'b','filled')
hold on 
line(freqqs(1:15),unwrap(angle(mod_resp)))
set(gca,'Xscale','log')
%% Yaw to start 
[pre, post] = get_all_tf_stats(Hi_yaw, Hf_yaw, freqqs, 8);
g = pre.mg .* 100;
p = pre.mp;

H_mean = g .* exp(1i.*p);

H = idfrd(H_mean(1:15),freqqs(1:15),1/10000);


sys = tfest(H,3,0);
mod_resp = squeeze(freqresp(sys,freqqs(1:15)));
% plot 
figure; 
subplot(2,1,1);
title('Third order mean Moths Yaw')
scatter(freqqs(1:15),abs(H_mean(1:15))./100,color="blue")
hold on 
line(freqqs(1:15),abs(mod_resp)./100,color="blue")
set(gca,'XScale','log')
set(gca,'Yscale','log')
title('Third order mean Moths Yaw')
x
subplot(2,1,2);
scatter(freqqs(1:15),unwrap(angle(H_mean(1:15))),color="blue")
hold on 
line(freqqs(1:15),unwrap(angle(mod_resp)),color="blue")
set(gca,'XScale','log')

%% Unwrapping Function 
function y = unwrap_minus_pi(x)
%UNWRAP_MINUS_PI Unwrap phase by subtracting 2*pi only (no additions)


    y = x;

    dx = diff(x);

    % Logical mask: only positive jumps beyond +pi
    jumps = dx > pi;

    % Cumulative correction
    correction = -2*pi * cumsum(jumps);

    % Apply correction starting from second element
    y(2:end) = y(2:end) + correction;
end
%% tf_stats
function [mg, low, high, mp, std_p] = tf_stats(x, unwrapped_angles)
% METHODS TAKEN FROM ROTH et al. PNAS 2016
%
% Compute mean and std for gain and phase of a complex transfer function x
%
% Inputs:
%   x                : vector of complex values
%   unwrapped_angles : vector of phase angles (radians)
%
% Returns:
%   mg     : mean gain
%   low    : lower gain bound
%   high   : upper gain bound
%   mp     : circular mean phase
%   std_p  : circular phase std

    N = length(x);

    % Gain statistics (log space)

    mags = abs(x);
    log_g = log(mags);

    l_m_G = mean(log_g);

    l_std_g = sqrt( (1/(N-1)) * sum( (log_g - l_m_G).^2 ) );

    % Convert back from log space
    mg   = exp(l_m_G);
    high = exp(l_m_G + l_std_g);
    low  = exp(l_m_G - l_std_g);

    % Phase statistics (circular)

    mp = angle(mean(exp(1i * unwrapped_angles)));

    R = abs(mean(exp(1i * unwrapped_angles)));

    std_p = sqrt(-2 * log(R));

end
%% Run all tf stats 
function [pre, post] = get_all_tf_stats(low, high, freqs, freq_max)
% Take in low-mass and high-mass matrices of complex transfer functions
% Rows = frequencies
% Columns = moths
%
% Returns two tables: pre (low) and post (high)

    if nargin < 4
        freq_max = 20;
    end

    %%Frequency filtering

    mask  = freqs <= freq_max;
    freqs = freqs(mask);

    low  = low(mask, :);
    high = high(mask, :);

    % Preallocate angle matrices

    l_angles = zeros(size(low));
    h_angles = zeros(size(high));

    % Unwrap phase (subtract-only version)

    for i = 1:size(low,2)
        l_angles(:,i) = unwrap_minus_pi(angle(low(:,i)));
        h_angles(:,i) = unwrap_minus_pi(angle(high(:,i)));
    end

    % Preallocate result arrays

    nF = length(freqs);

    pre_freq  = zeros(nF,1);
    pre_mg    = zeros(nF,1);
    pre_glow  = zeros(nF,1);
    pre_ghigh = zeros(nF,1);
    pre_mp    = zeros(nF,1);
    pre_stp   = zeros(nF,1);

    post_freq  = zeros(nF,1);
    post_mg    = zeros(nF,1);
    post_glow  = zeros(nF,1);
    post_ghigh = zeros(nF,1);
    post_mp    = zeros(nF,1);
    post_stp   = zeros(nF,1);

    % Main loop over frequencies

    for i = 1:nF

        f = freqs(i);

        % ---- LOW MASS ----
        [mg, glow, ghigh, mp, p_std] = tf_stats(low(i,:), l_angles(i,:));

        pre_freq(i)  = f;
        pre_mg(i)    = mg;
        pre_glow(i)  = glow;
        pre_ghigh(i) = ghigh;
        pre_mp(i)    = mp;
        pre_stp(i)   = p_std;

        % ---- HIGH MASS ----
        [mg, glow, ghigh, mp, p_std] = tf_stats(high(i,:), h_angles(i,:));

        post_freq(i)  = f;
        post_mg(i)    = mg;
        post_glow(i)  = glow;
        post_ghigh(i) = ghigh;
        post_mp(i)    = mp;
        post_stp(i)   = p_std;
    end

    %% Unwrap mean phase across frequency

    pre_mp  = unwrap_minus_pi(pre_mp);
    post_mp = unwrap_minus_pi(post_mp);

    %Standard error scaling (like Julia version)

    nLow  = size(low,2);
    nHigh = size(high,2);

    pre_glow  = pre_mg  - (pre_mg  - pre_glow)  ./ sqrt(nLow);
    pre_ghigh = pre_mg  + (pre_ghigh - pre_mg)  ./ sqrt(nLow);

    post_glow  = post_mg  - (post_mg - post_glow) ./ sqrt(nHigh);
    post_ghigh = post_mg  + (post_ghigh - post_mg) ./ sqrt(nHigh);

    pre_stp  = pre_stp  ./ sqrt(nLow);
    post_stp = post_stp ./ sqrt(nHigh);

    % Convert to tables (Julia DataFrame equivalent)

    pre = table(pre_freq, pre_mg, pre_glow, pre_ghigh, pre_mp, pre_stp, ...
                'VariableNames', {'freq','mg','glow','ghigh','mp','stp'});

    post = table(post_freq, post_mg, post_glow, post_ghigh, post_mp, post_stp, ...
                 'VariableNames', {'freq','mg','glow','ghigh','mp','stp'});
end
