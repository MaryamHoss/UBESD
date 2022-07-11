% size=2082
% ssnr_pred=zeros(size,1);
% ssnr_noisy=zeros(size,1);
% for i=1:size
% ssnr_pred(i)=segsnr( clean(i,:)/norm(clean(i,:)), prediction(i,:)/norm(prediction(i,:)), 14700 );
% i
% end
% for i=1:size
% ssnr_noisy(i)=segsnr( clean(i,:)/norm(clean(i,:)), noisy(i,:)/norm(noisy(i,:)), 14700 );
% i
% end
% vals=[ssnr_noisy,ssnr_pred];
% boxplot(vals)
% 
% 
% stoi_pred=zeros(size,1);
% stoi_noisy=zeros(size,1);
% 
% for i=1:size
% stoi_pred(i)=stoi( clean(i,:), double(prediction(i,:)), 14700 );
% i
% end
% 
% for i=1:size
% stoi_noisy(i)=stoi( clean(i,:), noisy(i,:), 14700 );
% i
% end
% vals=[stoi_noisy,stoi_pred];
% boxplot(vals)
clear
clc
%exp_Folder='experiments/speaker_specific_new/1'

exp_Folder='experiments/test'
exp_type= 'WithSpikes'  %'noSpikes'%
fwsegSNR=[]
segSNR=[]
d=[];
g=[];
xc=[];
xc_non=[];

acc=[];
subjects=[];

i=0;
for subject= 1:33
    subject
    for batch=0:89
        
        batch

        orig_file_1=[exp_Folder '/clean_' exp_type '_b' num2str(batch) '_s' num2str(subject) '.wav']
        orig_file_non=[exp_Folder '/clean_' exp_type '_b' num2str(batch) '_s' num2str(subject) '_gtest_unattended.wav']
        estimateFile = [exp_Folder '/prediction_' exp_type '_b' num2str(batch) '_s' num2str(subject) '_gtest.wav'];

        
        if exist(orig_file_1, 'file') == 0 || exist(estimateFile, 'file') == 0  || exist(orig_file_non, 'file') == 0  
          % File does not exist
          % Skip to bottom of loop and continue with the loop
          continue;
        end 
        if exist(orig_file_1, 'file') & exist(estimateFile, 'file') & exist(orig_file_non, 'file') 
            i=i+1;
            subjects(i)=subject;
        end
        orig=audioread(orig_file_1);
        orig_non=audioread(orig_file_non);
        estimate=audioread(estimateFile);
        %fsnr=comp_fwseg( orig_file_1, estimateFile );
        %ssnr=segsnr( orig/norm(orig), estimate/norm(estimate), 14700 );
        %%estimate/norm(estimate)
        %fwsegSNR=[fwsegSNR;fsnr];
        %segSNR=[segSNR;ssnr];
        %[d_i,g_i,rr,ss]=v_sigalign(estimate,orig,14700);
        %xco_i=sum(rr.*ss)/sqrt(sum(rr.^2)*sum(ss.^2));
        xco_i=sum(orig.*estimate)/sqrt(sum(orig.^2)*sum(estimate.^2));
        xco_i_non=sum(orig_non.*estimate)/sqrt(sum(orig_non.^2)*sum(estimate.^2));
        %if xco_i>0.5
            %acc_i=1;
        %else
            %acc_i=0;
        %end
        if xco_i>xco_i_non
            acc_i=1;
        else
            acc_i=0;
        end            
        %d=[d;d_i];
        %g=[g;g_i];
        xc=[xc;xco_i];
        xc_non=[xc_non;xco_i_non];
        acc=[acc;acc_i]

    end
end


%save_path=[exp_Folder '/fwsegSNR_results_WithSpikes.mat'];
%save(save_path,'fwsegSNR')


%save_path=[exp_Folder '/segSNR_results_WithSpikes.mat'];
%save(save_path,'segSNR')

save_path=[exp_Folder '/delay_correlation.mat'];
%save(save_path,'d','g','xc','acc')
save(save_path,'xc','xc_non','acc')

save_path=[exp_Folder '/subjects.mat'];
save(save_path,'subjects')

