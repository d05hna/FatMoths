load("/home/doshna/Documents/PHD/FatMoths/Steven/moths_v73.mat")
%% first try to fit for one moth one trial 
freqqs =  [0.2000,0.3000,0.5000,0.7000,1.100,1.300,1.700,1.900,2.300,2.900,3.700,4.300,5.300,6.100,7.900,8.900,11.30,13.70];

m1hi = hi(:,1);
H = idfrd(m1hi,freqqs,1/100);
sys = tfest(H,2,1);
mod_resp = squeeze(freqresp(sys,freqqs));

%%
figure; 
subplot(2,1,1)
scatter(freqqs,abs(m1hi),'b','filled');
hold on 
line(freqqs,abs(mod_resp))
set(gca,'XScale','log')
set(gca,'Yscale','log')
hold off
subplot(2,1,2)
scatter(freqqs,unwrap(angle(m1hi)),'b','filled')
hold on 
line(freqqs,angle(mod_resp))
set(gca,'Xscale','log')
%% Mean Free Flight 

meani = mean(hi,2);
H = idfrd(meani,freqqs,1/100);
sys = tfest(H,2,1);mod_resp = squeeze(freqresp(sys,freqqs));
figure; 
subplot(2,1,1)
scatter(freqqs,abs(meani),'b','filled');
hold on 
line(freqqs,abs(mod_resp))
set(gca,'XScale','log')
set(gca,'Yscale','log')
hold off
subplot(2,1,2)
scatter(freqqs,unwrap(angle(meani)),'b','filled')
hold on 
line(freqqs,unwrap(angle(mod_resp)))
set(gca,'Xscale','log')