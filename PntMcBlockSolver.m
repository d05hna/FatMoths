function [Ci, Cf, Hhatf]=PntMcBlockSolver(Hi,Hf,mi,mf)


    % solves for C block of Flora feedback loop assuming a simple inertial
    % mechanical model (point mass)
    
    % in this case, the mechanics model is simple inertial m=1/(ms^2)
    
    % Hi is the peak freq response of the initial time bucket
    % Hf is the peak freq response of the 40-60 sec time bucket
    % mi is initial moth mass
    % mf is the regressed mass at 60 seconds
    
    % Ci is the initial sensory processing block response
    % Hhatf is the predicted response for the 40-60 sec time bucket given Ci and the final mass
    
    freqs = [0.2000    0.3000    0.5000    0.7000    1.1000    1.3000...
        1.7000    1.9000    2.3000    2.9000    3.7000    4.3000    5.3000...
        6.1000  7.9000    8.9000   11.3000   13.7000];
    
    s = 1i*2*pi*freqs';
    Mi = 1./(mi*(s.^2));
    Mf = 1./(mf*(s.^2));


%     b=1;


%     bi = mi;
%     bf = mf;


%     Mi = 1./(mi*(s.^2)+bi*s);
%     Mf = 1./(mf*(s.^2)+bf*s);
    Ci = Hi./(Mi-Hi.*Mi);
    Hhatf = (Ci.*Mf)./(1+Ci.*Mf);
    
    
    Cf = Hf./(Mf-Hf.*Mf);
    
    
         %C Block plots
    figure
    ax = subplot(2,1,1);
    plot(freqs,abs(Ci));
    hold on;
    plot(freqs,abs(Cf));
    set(ax,'XScale','log','YScale','log'); 
    ylabel('Gain');
    title(['C Block'])
    legend('Initial Controller','Final Controller');
    
    ax = subplot(2,1,2);
    plot(freqs,(180/pi)*angle(Ci));
    hold on;
    plot(freqs,(180/pi)*angle(Cf));
    set(ax,'XScale','log');
    xlabel('Frequency (Hz)');
    ylabel('Phase Difference (degrees)');






    %Hf vs Hfhat plots
    figure
    subplot(2,1,1);
    semilogx(freqs,log10(abs(Hf)));
    hold on;
    semilogx(freqs,log10(abs(Hi)));
    semilogx(freqs,log10(abs(Hhatf)));
    legend('Measured Final Response','Initial Response', 'Predicted Final Response');
    ylabel('Log10 of Positional Gain');
    %title(tit)
    
    subplot(2,1,2);
    semilogx(freqs,(180/pi)*angle(Hf));
    hold on;
    semilogx(freqs,(180/pi)*angle(Hi));
    semilogx(freqs,(180/pi)*angle(Hhatf));
    xlabel('Frequency (Hz)');
    ylabel('Phase Difference (degrees)');


    
end

