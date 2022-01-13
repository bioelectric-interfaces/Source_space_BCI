#!/usr/bin/env python3
# -*- coding: utf-8 -*-ы
"""
Created on Fri Jun 18 18:56:32 2021

@author: gurasog

"""
import mne
from sklearn.base import BaseEstimator,TransformerMixin
from scipy.signal import butter, lfilter, savgol_coeffs,filtfilt
from scipy import signal
from sklearn.feature_selection import mutual_info_classif
import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import ShuffleSplit, cross_val_score
from mne import Epochs, pick_types, events_from_annotations
from mne.channels import make_standard_montage
from mne.io import concatenate_raws, read_raw_edf
from mne.datasets import eegbci
from mne.decoding import CSP
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV 
from  sklearn.metrics import accuracy_score
import os
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import nibabel
from nilearn.plotting import plot_glass_brain
import numpy as np
from mne.channels import compute_native_head_t, read_custom_montage
from mne.viz import plot_alignment
from scipy.signal import hilbert
import pandas as pd
import os.path as op
from mne.datasets import eegbci
from mne.datasets import fetch_fsaverage
from mne.minimum_norm.inverse import prepare_inverse_operator, _assemble_kernel
from scipy.signal import butter, lfilter, savgol_coeffs,filtfilt
from scipy import signal
from sklearn.feature_selection import mutual_info_classif

from mne.minimum_norm import make_inverse_operator, apply_inverse,apply_inverse_raw, apply_inverse_epochs

from sklearn.svm import SVC

class SourceSpaceEsimator_2(BaseEstimator):
    def __init__(self, param, fwd, cov, subject, subjects_dir,info,clusters,inv,labels_i_need_names=None,clusters_num=30,lamb=1, random_state=None):
        self.param=param
        self.fwd=fwd
        self.cov=cov
        self.subject=subject
        self.subjects_dir=subjects_dir
        self.info=info
        self.clusters=clusters
        self.inv=inv
        self.labels_i_need_names=labels_i_need_names
        self.clusters_num=clusters_num
        self.lamb=lamb
        self.random_state = random_state
        self.svc=None
        
        print('lol')
  
    
    def get_params(self, deep=True):
        return {"param": self.param,
                "fwd": self.fwd,
                "cov":self.cov,
                "subject":self.subject,
                "subjects_dir":self.subjects_dir,
                "info":self.info,
                "clusters":self.clusters,
                "inv":self.inv,
                "labels_i_need_names":self.labels_i_need_names,
                "clusters_num":self.clusters_num,
                "lamb":self.lamb,
                "random_state": self.random_state
               }

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    
    def predict(self,X,y=None,xy_flg=True):
        

        epochs=X
        epochs_data=np.array([self.create_source_data(epochs[i].copy(), self.fwd,self.inv,fs=160,mode='full' ) for i in range(len(epochs))])
        #print(epochs_data.shape)
        #epochs_data_train=[create_source_data(epochs_train[i].copy(), fwd,inv,fs=160 ) for i in range(len(epochs_train))]
        epochs_data_train=epochs_data#[:,:,160:-160]
        #print(epochs_data_train.shape)
        epochs_data_train=np.array( [np.mean(epochs_data_train[:,np.where(self.clusters==i)[0],:],1) for i in range(20)] )
        #print(epochs_data_train.shape)
        epochs_data_train=epochs_data_train.transpose(1,0,2)

        if xy_flg:
            a,b=self.make_training_set(np.array(epochs_data_train),epochs.events, xy_flg=True)
            X_means=np.mean(a,2)
            Y=b
            #return X_means, Y

        else: 
            b=self.make_training_set(np.array(epochs_data_train),epochs.events,xy_flg=False)
            Y=b
            #return Y


        print('Use trained SVM')
        return self.svc.predict(X_means), Y

        
    
    def make_training_set(self, epochs_data_train,events, concat_flg=True,xy_flg=True):

        if xy_flg:
            X_all=[]
            Y_all=[]
            for i in range(len(epochs_data_train)):
                list_of_states_i, train_data_list_i=self.make_data(epochs_data_train[i],320,50,[events[i,2]]*epochs_data_train[i].shape[1])
                
                X_=[]

                for state_i in range(len(list_of_states_i)-1):
                    #X_.append(np.mean(epochs_data_train[i][:,train_data_list_i[state_i]],1))
                    X_.append(epochs_data_train[i][:,train_data_list_i[state_i]])

                X_all.append(X_)
                Y_all.append(list_of_states_i[:-1])
            
            X_all=np.array(X_all)
            Y_all=np.array(Y_all)
            
            if concat_flg:
                X_all=np.concatenate(X_all)
                Y_all=np.concatenate(Y_all)

            return X_all,Y_all

        else:

            Y_all=[]
            for i in range(len(epochs_data_train)):
                list_of_states_i, train_data_list_i=self.make_data(epochs_data_train[i],320,50,[events[i,2]]*epochs_data_train[i].shape[1])
                Y_all.append(list_of_states_i[:-1])
            
            
            Y_all=np.array(Y_all)
            
            if concat_flg:
                
                Y_all=np.concatenate(Y_all)

            return Y_all


    def make_data(self, raw,window,bias,states,Fs=160):

        try:
        
            time_samples=len(raw['Fz'][0][0])
        
        except :
            
            time_samples=len(np.arange(0,raw.shape[1]*1/Fs,1/Fs))
            
            
        list_of_states=[]
        train_data_list=[]

        for c in np.arange(0,(time_samples-(window-bias)),bias):

                sum_of_states=sum(states[c:c+window])

                if (sum_of_states%window)==0 and (sum_of_states/window)==states[c]:

                    list_of_states.append(states[c])

                    train_data_list.append(np.arange(c,(c+window)))


        return list_of_states, train_data_list


    def fit(self,X,y=None):
        
        epochs=X[:32]
        epochs_data=np.array([self.create_source_data(epochs[i].copy(), self.fwd,self.inv,fs=160,mode='full' ) for i in range(len(epochs))])
        #print(epochs_data.shape)
        #epochs_data_train=[create_source_data(epochs_train[i].copy(), fwd,inv,fs=160 ) for i in range(len(epochs_train))]
        epochs_data_train=epochs_data#[:,:,160:-160]
        #print(epochs_data_train.shape)
        epochs_data_train=np.array( [np.mean(epochs_data_train[:,np.where(self.clusters==i)[0],:],1) for i in range(20)] )
        #print(epochs_data_train.shape)
        epochs_data_train=epochs_data_train.transpose(1,0,2)

        
        a,b=self.make_training_set(np.array(epochs_data_train),epochs.events, xy_flg=True)
        X_means=np.mean(a,2)
        Y=b
        #return X_means, Y


        print('Set the SVM')
        self.svc=SVC(random_state=123)
        self.svc.fit(X_means, Y)
        #return self.svc.predict(X_means), Y
        return self
    
    def all_about_labels(self,fwd,subject_dir='fsaverage',subject='fsaverage',labels_i_need_names=['postcentral-lh','precentral-lh','postcentral-rh','precentral-rh','paracentral-lh', 'paracentral-rh','caudalmiddlefrontal-lh', 'caudalmiddlefrontal-rh']):

        fs_dir = fetch_fsaverage(verbose=True)
        subject_dir = op.dirname(fs_dir)
        
        labels=mne.read_labels_from_annot(subject, 
                                   parc='aparc', 
                                   hemi='both', 
                                   surf_name='pial', 
                                   annot_fname=None, 
                                   regexp=None, 
                                   subjects_dir=subject_dir, 
                                   #sort=True, 
                                   verbose=None)

        left_hemi,right_hemi=fwd["src"]

        sources_idx=np.r_[left_hemi["vertno"],right_hemi["vertno"]+len(left_hemi["vertno"])]
        print(sources_idx)
        sources_idx=left_hemi["vertno"]
        vertices_i_need=[]
        labels_i_need=[]
        poses=np.zeros((0,3))
        
        for i in range(len(labels)):
            labels[i].forward_vertices=np.where(np.isin(sources_idx,labels[i].vertices))[0]
            if labels[i].name in labels_i_need_names:
                labels_i_need.append(labels[i])
                if labels[i].name[-2:]=='lh':
                    vertices_i_need=np.concatenate([vertices_i_need,labels[i].forward_vertices])
                    inds_in_this_label=[labels[i].vertices.tolist().index(k) for k in labels[i].forward_vertices]
                    poses=np.concatenate([poses,labels[i].pos[inds_in_this_label,:]])

                elif labels[i].name[-2:]=='rh':
                    vertices_i_need=np.concatenate([vertices_i_need,labels[i].forward_vertices+fwd['source_rr'].shape[0]/2])
                    inds_in_this_label=[labels[i].vertices.tolist().index(k) for k in labels[i].forward_vertices ]
                    poses=np.concatenate([poses,labels[i].pos[inds_in_this_label,:]])

        vertices_i_need_three_orient=self.threeplet(vertices_i_need)
        vertices_i_need_three_orient=np.array(vertices_i_need_three_orient).astype(int)

        return vertices_i_need_three_orient

    

    def make_forward_for_fsaverage(self,raw):
        subject = 'fsaverage'
        trans = 'fsaverage'  # MNE has a built-in fsaverage transformation
        src = op.join(fs_dir, 'bem', 'fsaverage-ico-5-src.fif')
        bem = op.join(fs_dir, 'bem', 'fsaverage-5120-5120-5120-bem-sol.fif')
        
        
        raw.set_eeg_reference(projection=True)
        events, _ = mne.events_from_annotations(raw, event_id=dict(T1=2, T2=3))
    
        cov = mne.compute_raw_covariance(raw)
    
        fwd = mne.make_forward_solution(raw.info, trans=trans, src=src,
                                        bem=bem, eeg=True, mindist=5.0, n_jobs=1)
        #print(fwd)
    
        # Use fwd to compute the sensitivity map for illustration purposes
        
        cov = mne.compute_raw_covariance(raw)
    
        
        
        return raw,cov,fwd
    
    def threeplet(self,vertices_i_need):
        vertices_i_need_three=[]
        
        for i in range(len(vertices_i_need)):
            vertices_i_need_three.append(vertices_i_need[i]*3)
            vertices_i_need_three.append(vertices_i_need[i]*3+1)
            vertices_i_need_three.append(vertices_i_need[i]*3+2)
            
        return vertices_i_need_three
    
    def return_gain_matrix(self,fwd):
        gain_matrix=fwd['sol']['data']
        return gain_matrix
    
    def return_gain_matrix_ROI(self,fwd,vertices_i_need_three_orient):

        gain_matrix=self.return_gain_matrix(fwd)
        gain_matrix_small=gain_matrix[:,vertices_i_need_three_orient]
    
    
        return gain_matrix_small
    
    def clusterization_of_gain(self,gain_matrix_small,cluster_num=30):
        X_inds=np.arange(0,gain_matrix_small.shape[1],3)
        Y_inds=X_inds+1
        Z_inds=X_inds+2
    
        gain_X=gain_matrix_small[:,X_inds]
        gain_Y=gain_matrix_small[:,Y_inds]
        gain_Z=gain_matrix_small[:,Z_inds]
    
        gain_all=np.concatenate([gain_X,gain_Y,gain_Z]).T
        
        km=KMeans(cluster_num)
        #km = AgglomerativeClustering(cluster_num)
        
        km.fit(gain_all)
    
        cluster_ids=km.fit_predict(gain_all)
        
        return cluster_ids
    
    def extract_inverse_MNE_ROI(self,info,fwd,cov,vertices_i_need_three_orient,lambda2=1):
    #TODO: add other types of inverse
    
        inv = mne.minimum_norm.make_inverse_operator( self.info, fwd, cov, verbose=True)
        inv=prepare_inverse_operator(inv,1,lambda2=lambda2,method='MNE')
        K, noise_norm,vertno,source_nn=_assemble_kernel(inv,None,'MNE',None)
    
        inverse_matrix=K
        inverse_matrix_small=inverse_matrix[vertices_i_need_three_orient,:]
        
        return inverse_matrix_small
    
    def create_source_data_for_test(self,raw,fwd,inverse_matrix_small,cluster_ids,cluster_num=30,fs=0):
        if type(raw) is np.ndarray:
            raw_matrix=raw
            fs=fs
        else:
            raw_matrix=raw.get_data()
            fs=raw.info['sfreq']
        
        alpha_raw=self.filter_data(raw_matrix,8,12,fs)
        source_alpha=self.extract_inverse(inverse_matrix_small,alpha_raw)
        del alpha_raw
        alpha_env=self.extract_envelope(source_alpha)
        del source_alpha
        alpha_cl=self.cluster_band(alpha_env,cluster_ids,cluster_num)
        del alpha_env
        
        beta_raw=self.filter_data(raw_matrix,15,25,fs)
        source_beta=self.extract_inverse(inverse_matrix_small,beta_raw)
        del beta_raw
        beta_env=self.extract_envelope(source_beta)
        del source_beta
        beta_cl=self.cluster_band(beta_env,cluster_ids,cluster_num)
        del beta_env
        
        theta_raw=self.filter_data(raw_matrix,2,7,fs)
        source_theta=self.extract_inverse(inverse_matrix_small,theta_raw)
        del theta_raw
        theta_env=self.extract_envelope(source_theta)
        del source_theta
        theta_cl=self.cluster_band(theta_env,cluster_ids,cluster_num)
        del theta_env
    
        clusters_data=np.concatenate([alpha_cl,beta_cl,theta_cl])
        
        return clusters_data
    
    def extract_labels(self,subject,subjects_dir):
        
        labels=mne.read_labels_from_annot(subject,  parc='aparc',   hemi='both', surf_name='pial', 
                                   annot_fname=None,  regexp=None,  subjects_dir=subjects_dir,   verbose=None)
        
        return [labels[44],labels[45],labels[48],labels[49]]
    
    # def create_source_data(self,raw,fwd,inv,cluster_ids,labels,fs=0):
        
    #     labels=self.extract_labels(self.subject,self.subjects_dir)
        
    #     if type(raw) is np.ndarray:
    #         raw_matrix=raw
    #         fs=fs
    #     else:
    #         raw_matrix=raw.get_data()
    #         fs=raw.info['sfreq']
        
    #     alpha_raw=self.filter_data(raw_matrix,8,12,fs, verbose=False)
    #     alpha_raw=alpha_raw.apply_hilbert()
    #     betta_raw=self.filter_data(raw_matrix,15,25,fs, verbose=False)
    #     betta_raw=betta_raw.apply_hilbert()

    #     for i in range(labels):
    #         label=labels[i]
    #         source_alpha= abs(apply_inverse_raw(alpha_raw, inv, lambda2=0.01, method= 'MNE', label=label, verbose=False).data)
    #         source_betta= abs(apply_inverse_raw(betta_raw, inv, lambda2=0.01, method= 'MNE', label=label, verbose=False).data)
        
    #         sources_list.append(source_alpha)
    #         sources_list.append(source_betta)
            
    #     clusters_data=np.concatenate(sources_list)
        
    #     return clusters_data

    def combine_data_in_source(self,source_alpha):
        matr=np.zeros((source_alpha.shape[0]//3,source_alpha.shape[1]))
        #(matr.shape)
        for i in range(source_alpha.shape[0]//3):
            matr[i,:]=np.sqrt(source_alpha[i*3,:]*source_alpha[i*3,:]+
                              source_alpha[i*3+1,:]*source_alpha[i*3+1,:]+
                            source_alpha[i*3+2,:]*source_alpha[i*3+2,:])
        #print(matr.shape)
        return matr
    
    def filter_data(self,eeg_data,low,high,fs,order=4):
        chunk=eeg_data
        n_channels=eeg_data.shape[0]
        b, a = butter(order, [low/fs*2, high/fs*2], btype='band')
        zi = np.zeros((max(len(b), len(a)) - 1, n_channels))
        #y, zi = lfilter(b, a, chunk.T, axis=0, zi=zi)
        y = filtfilt(b, a, chunk.T, axis=0)
        return y.T
    
    def extract_envelope(self,signal_raw_):
        signal_=hilbert(signal_raw_)
        env_=np.abs(signal_)

        return env_
    
    def extract_inverse(self,inverse_matrix,data):
        source=np.matmul(inverse_matrix,data)
        return source
    
    
    def logarifming(self,X):
    
        #assert (X.shape[0]>X.shape[1])
        return np.log(X)
    
    def scaling(self,x_train,x_test):

        '''
        returns scaled data or scale to data be scaled
        '''

        if x_test is None:
            scaler = StandardScaler()
            scaler.fit(x_train)
            x_train_new=scaler.transform(x_train)
            #x_test_new=scaler.transform(x_test)

            return x_train_new,scaler

        else:
            scaler = StandardScaler()
            scaler.fit(x_train)
            x_train_new=scaler.transform(x_train)
            x_test_new=scaler.transform(x_test)

            return x_train_new,x_test_new


    def cluster_band(self, source_alpha,cluster_ids,cluster_num=30):
        
        matr=self.combine_data_in_source(source_alpha)
        clusters_data=self.compute_cluster_mean(matr,cluster_ids,cluster_num)

        return clusters_data


    def compute_cluster_mean(self,matr,cluster_ids,cluster_num=30):
        
        clusters_data=np.zeros((cluster_num,matr.shape[1]))
        for i in range(cluster_num):
            cl_ind=np.where(cluster_ids==i)[0]

            clusters_data[i,:]=np.mean(matr[cl_ind,:],0)

        return clusters_data


    #def extract_labels(self,subject,subjects_dir):
    #    labels=mne.read_labels_from_annot(subject,  parc='aparc',   hemi='both', surf_name='pial', 
    #                               annot_fname=None,  regexp=None,  subjects_dir=subjects_dir,   verbose=None)

    #    return [labels[44],labels[45],labels[48],labels[49]]


    def extract_labels(self,subject,subjects_dir,selected_labels=['postcentral-lh','postcentral-rh', 'paracentral-lh',  'paracentral-rh']):
        '''


        '''

        labels=mne.read_labels_from_annot(subject,  parc='aparc',   hemi='both', surf_name='pial', 
                                   annot_fname=None,  regexp=None,  subjects_dir=subjects_dir,   verbose=None)
        
        labels_names=[labels[i].name for i in range(len(labels))]
        
        inds=[labels_names.index(selected_labels[i]) for i in range(len(selected_labels))]
        
        return [labels[i] for i in inds]

    def create_source_data(self,raw,fwd,inv,fs=0,mode='mean'):
        
        # Сейчас здесь только усреднение
        '''
        mode: describes data in which sense will be sent back
            'mean' -
            'full' -
        '''
        
        
        fs_dir = fetch_fsaverage(verbose=False)
        subjects_dir = op.dirname(fs_dir)
        subject='fsaverage'

        labels=self.extract_labels(subject,subjects_dir)
        
        print(labels)
        if type(raw) is np.ndarray:
            raw_matrix=raw
            fs=fs
        else:
            raw_matrix=raw.get_data()
            fs=raw.info['sfreq']

        alpha_raw=raw.copy().filter(8,12,phase='zero', verbose=False)
        alpha_raw=alpha_raw.apply_hilbert()

        betta_raw=raw.copy().filter(15,25,phase='zero', verbose=False)
        betta_raw=betta_raw.apply_hilbert()


        sources_list=[]
        for i in range(len(labels)):
            label=labels[i]
            #source_alpha= np.mean(abs(apply_inverse_raw(alpha_raw, inv, lambda2=0.01, method= 'MNE', label=label).data),0)
            #source_betta= np.mean(abs(apply_inverse_raw(betta_raw, inv, lambda2=0.01, method= 'MNE', label=label).data),0)
            
            if mode=='mean':
                source_alpha= np.mean(abs(apply_inverse_epochs(alpha_raw, inv, lambda2=self.lamb, method= 'sLORETA', label=label, verbose=False)[0].data),0)
                source_betta= np.mean(abs(apply_inverse_epochs(betta_raw, inv, lambda2=self.lamb, method= 'sLORETA', label=label, verbose=False)[0].data),0)

                sources_list.append(source_alpha)
                sources_list.append(source_betta)
                
            elif mode=='full':
                
                source_alpha= abs(apply_inverse_epochs(alpha_raw, inv, lambda2=0.1, method= 'sLORETA', label=label, verbose=False)[0].data)
                #source_betta= abs(apply_inverse_epochs(betta_raw, inv, lambda2=0.1, method= 'sLORETA', label=label)[0].data)

                sources_list.append(source_alpha)
                #sources_list.append(source_betta)
                
                

        clusters_data=np.vstack(sources_list)

        return clusters_data