 function [seg,glo]=v_snrseg(s,r,fs,m,tf)
 %V_SNRSEG Measure segmental and global SNR [SEG,GLO]=(S,R,FS,M,TF)
 %
 %Usage: (1) seg=v_snrseg(s,r,fs);                  % s & r are noisy and clean signal
 %       (2) seg=v_snrseg(s,r,fs,'wz');             % no VAD or inerpolation used ['Vq' is default]
 %       (3) [seg,glo]=v_snrseg(s,r,fs,'Vq',0.03);  % 30 ms frames
 %
 % Inputs:    s  test signal
 %            r  reference signal
 %           fs  sample frequency (Hz)
 %            m  mode [default = 'Vq']
 %                 w = No VAD - use whole file
 %                 v = use sohn VAD to discard silent portions
 %                 V = use P.56-based VAD to discard silent portions [default]
 %                 a = A-weight the signals
 %                 b = weight signals by BS-468
 %                 q = use quadratic interpolation to remove delays +- 1 sample
 %                 z = do not do any alignment
 %                 p = plot results
 %           tf  frame increment [0.01]
 %
 % Outputs: seg = Segmental SNR in dB
 %          glo = Global SNR in dB (typically 7 dB greater than SNR-seg)
 %

 if nargin<4 || ~ischar(m)
     m='Vq';
 end
 if nargin<5 || ~numel(tf)
     tf=0.01; % default frame length is 10 ms
 end
 snmax=100;  % clipping limit for SNR
 
 % filter the input signals if required
 
 if any(m=='a')  % A-weighting
     [b,a]=v_stdspectrum(2,'z',fs);
     s=filter(b,a,s);
     r=filter(b,a,r);
 elseif any(m=='b') %  BS-468 weighting
     [b,a]=v_stdspectrum(8,'z',fs);
     s=filter(b,a,s);
     r=filter(b,a,r);
 end
 
 mq=~any(m=='z');
 nr=min(length(r), length(s));
 kf=round(tf*fs); % length of frame in samples
 ifr=kf+mq:kf:nr-mq; % ending sample of each frame
 ifl=ifr(end);
 nf=numel(ifr); % number of frames
 if mq % perform interpolation
     ssm=reshape(s(2:ifl)-s(3:ifl+1),kf,nf);
     ssp=reshape(s(2:ifl)-s(1:ifl-1),kf,nf);
     sr=reshape(s(2:ifl)-r(2:ifl),kf,nf);
     am=min(max(sum(sr.*ssm,1)./sum(ssm.^2,1),0),1); %optimum dist between s(2:ifl) and s(3:ifl+1)
     ap=min(max(sum(sr.*ssp,1)./sum(ssp.^2,1),0),1); %optimum dist between s(2:ifl) and s(1:ifl-1)
     ef=min(sum((sr-repmat(am,kf,1).*ssm).^2,1),sum((sr-repmat(ap,kf,1).*ssp).^2,1)); % select the best for each frame
 else % no interpolation
     ef=sum(reshape((s(1:ifl)-r(1:ifl)).^2,kf,nf),1);
 end
 rf=sum(reshape(r(mq+1:ifl).^2,kf,nf),1);
 em=ef==0; % mask for zero noise frames
 rm=rf==0; % mask for zero reference frames
 snf=10*log10((rf+rm)./(ef+em));
 snf(rm)=-snmax;
 snf(em)=snmax;
 
 % select the frames to include
 
 if any(m=='w')
     vf=true(1,nf); % include all frames
 elseif any(m=='v')
     vs=v_vadsohn(r,fs,'na');
     nvs=length(vs);
     [vss,vix]=sort([ifr'; vs(:,2)]);
     vjx=zeros(nvs+nf,5);
     vjx(vix,1)=(1:nvs+nf)'; % sorted position
     vjx(1:nf,2)=vjx(1:nf,1)-(1:nf)'; % prev VAD frame end (or 0 or nvs+1 if none)
     vjx(nf+1:end,2)=vjx(nf+1:end,1)-(1:nvs)'; % prev snr frame end (or 0 or nvs+1 if none)
     dvs=[vss(1)-mq; vss(2:end)-vss(1:end-1)];  % number of samples from previous frame boundary
     vjx(:,3)=dvs(vjx(:,1)); % number of samples from previous frame boundary
     vjx(1:nf,4)=vs(min(1+vjx(1:nf,2),nvs),3); % VAD result for samples between prev frame boundary and this one
     vjx(nf+1:end,4)=vs(:,3); % VAD result for samples between prev frame boundary and this one
     vjx(1:nf,5)=1:nf; % SNR frame to accumulate into
     vjx(vjx(nf+1:end,2)>=nf,3)=0;  % zap any VAD frame beyond the last snr frame
     vjx(nf+1:end,5)=min(vjx(nf+1:end,2)+1,nf); % SNR frame to accumulate into
     vf=full(sparse(1,vjx(:,5),vjx(:,3).*vjx(:,4),1,nf))>kf/2; % accumulate into SNR frames and compare with threshold
 else  % default is 'V'
     [lev,af,fso,vad]=v_activlev(r,fs);    % do VAD on reference signal
     vf=sum(reshape(vad(mq+1:ifl),kf,nf),1)>kf/2; % find frames that are mostly active
 end
 seg=mean(snf(vf));
 glo=10*log10(sum(rf(vf))/sum(ef(vf)));
 
 if ~nargout || any (m=='p')
     subplot(311);
     plot((1:length(s))/fs,s);
     ylabel('Signal');
     title(sprintf('SNR = %.1f dB, SNR_{seg} = %.1f dB',glo,seg));
     axh(1)=gca;
     subplot(312);
     plot((1:length(r))/fs,r);
     ylabel('Reference');
     axh(2)=gca;
     subplot(313);
     snv=snf;
     snv(~vf)=NaN;
     snu=snf;
     snu(vf>0)=NaN;
     plot([1 nr]/fs,[glo seg; glo seg],':k',((1:nf)*kf+(1-kf)/2)/fs,snv,'-b',((1:nf)*kf+(1-kf)/2)/fs,snu,'-r');
     ylabel('Frame SNR');
     xlabel('Time (s)');
     axh(3)=gca;
     linkaxes(axh,'x');
 end