function [Ci, Cf, Hhatf, Sysi, Sysf]=LatLincBlockSolver(Hi,Hf,mi,mf,l,w,c,St,ni,nf,tit)

%function [Ci, Cf, Hhatf, sysi, sysf]=LatLincBlockSolver(Hi,Hf,mi,mf,l,w,c,St,ni,nf)
    %   calculates (1) initial and (2) final 'controller' and predicted  
    %   final response assuming a constant controller of continually  
    %   feeding hawkmoths assuming a body mechanics model based on the 
    %   linearized  equations of motion from Sun 2014 which are themselves 
    %   based on the coupled nonlinear ODEs from [Etkin and Reid 1996] as 
    %   well as [Taylor and Thomas 2003]. 
    
    %   driven frequencies
    freqs = [0.2000    0.3000    0.5000    0.7000    1.1000    1.3000...
        1.7000    1.9000    2.3000    2.9000    3.7000    4.3000    5.3000...
        6.1000  7.9000    8.9000   11.3000   13.7000];
    
    %   applying mechanics model: MBlockSolver based on linearize EOM from
    %   Sun: ouput is coefficients of transfer function polynomials for
    %   numerator (num_) and denominator (den_), as well as 
%     [numi, deni]=LatLinMBlockSolver(mi, l, w, ni, c, St, title);
%     
%     [numf, denf]=LatLinMBlockSolver(mf, l, w, nf, c, St, title);
    
    [numi, deni, systemi]=LatLinMBlockSolver(mi, l, w, ni, c, St, tit);
    
    [numf, denf, systemf]=LatLinMBlockSolver(mf, l, w, nf, c, St, tit);
    
    %   loops for generating transfer function numerator and denominator
    %   polynomials from coefficients
% % %     numis = 0;
% % %     for i=1:length(numi)
% % %         numis = numis + numi(i)*s.^(length(numi)-i);
% % %     end
% % %     
% % %     denis = 0;
% % %     for i=1:length(deni)
% % %         denis = denis + deni(i)*s.^(length(deni)-i);
% % %     end
% % %     
% % %     Mi = numis./denis;
% % %     
% % %     numfs = 0;
% % %     for i=1:length(numf)
% % %         numfs = numfs + numf(i)*s.^(length(numf)-i);
% % %     end
% % %     
% % %     denfs = 0;
% % %     for i=1:length(denf)
% % %         denfs = denfs + denf(i)*s.^(length(denf)-i);
% % %     end
% % %     
% % %     Mf = numfs./denfs;
    
    %M = 1/(mf.*s.^2);
    
    %Mb = (1/((mf.*s.^2)+(mf*s)))';
    
Sysi = tf(numi, deni);
Sysf = tf(numf, denf);

inputs = freqs.*(2*pi*1j);

Mi=freqresp(Sysi,inputs);
Mf=freqresp(Sysf,inputs);

Mi=reshape(Mi,length(Mi),1,1);
Mf=reshape(Mf,length(Mf),1,1);
    
    %   initial controller
    Ci = (inputs'.*Hi)./(Mi-Hi.*Mi);
    
    %   predicted final response for constant controller assumption
    Hhatf = (Ci.*Mf)./(inputs'+Ci.*Mf);
    
    %   final controller
    Cf = (inputs'.*Hf)./(Mf-Hf.*Mf);
    
%        %C Block plots
%     figure
%     ax = subplot(2,1,1);
%     plot(freqs,abs(Ci),'LineWidth',2);
%     hold on;
%     plot(freqs,abs(Cf),'LineWidth',2);
%     set(ax,'XScale','log','YScale','log'); 
%     ylabel('Gain');
%     %title(['C Block'])
%     %legend('Initial Controller','Final Controller');
%     axis([0.18 15 10^5 10^10]);
%     
%     ax = subplot(2,1,2);
%     plot(freqs,(180/pi)*angle(Ci),'LineWidth',2);
%     hold on;
%     plot(freqs,(180/pi)*angle(Cf),'LineWidth',2);
%     set(ax,'XScale','log');
%     xlabel('Frequency (Hz)');
%     ylabel('Phase Difference (degrees)');
%     axis([0.18 15 -185 185]);



%     %Hf vs Hfhat plots
%     figure
%     subplot(2,1,1);
%     semilogx(freqs,log10(abs(Hf)),'r','LineWidth',2);
%     hold on;
%     semilogx(freqs,log10(abs(Hi)),'b','LineWidth',2);
%     semilogx(freqs,log10(abs(Hhatf)),'k','LineWidth',2);
%     %legend('Measured Final Response','Predicted Final Response','Initial Response');
%     ylabel('Log10 of Positional Gain');
%     axis([0.18 15 -1.5 0.5]);
%     
%     subplot(2,1,2);
%     semilogx(freqs,(180/pi)*angle(Hf),'r','LineWidth',2);
%     hold on;
%     semilogx(freqs,(180/pi)*angle(Hi),'b','LineWidth',2);
%     semilogx(freqs,(180/pi)*angle(Hhatf),'k','LineWidth',2);
%     xlabel('Frequency (Hz)');
%     ylabel('Phase Difference (degrees)');
%     axis([0.18 15 -185 185]);
    
    
%     % plots Mechanics Blocks
%     figure
%     subplot(2,1,1);
%     semilogx(freqs,log10(abs(Mf)));
%     hold on;
%     semilogx(freqs,log10(abs(M)));
%     semilogx(freqs,log10(abs(Mb)));
%     legend('Linearized State Space Mechanics','Undamped Mass Mechanics','Damped Mass Mechanics');
%     ylabel('Log10 of Gain');
%     title('Initial, Predicted Final, and Measured Final Response')
%     
%     subplot(2,1,2);
%     semilogx(freqs,(180/pi)*angle(Mf));
%     hold on;
%     semilogx(freqs,(180/pi)*angle(M));
%     semilogx(freqs,(180/pi)*angle(Mb));
%     xlabel('Frequency (Hz)');
%     ylabel('Phase Difference (degrees)');
    
end


    