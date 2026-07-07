#changes made:
#rank-deficieny: swap pearson for ledoit wolf and make 200x200 stable from 16-vol task time
#forced relation learning: kill diagonal and fisher z shift
#group sub batch: read sick v healthy from clinical csv. wrap 7 task graphs
#sentivity testing:chop graphs into density boxes


#oh and imports were missing
import os
import numpy as np
import pandas as pd
from pathlib import Path
from nilearn.input_data import NiftiLabelsMasker
from nilearn import NiftiLabelsMasker
from nilearn.connectome import ConnectivityMeasure
from sklearn.covariance import LedoitWolf
import torch
from torch_geometric.data import Data


class preprocessBOLD:
  def __init__(self):
    self.datafolder = "../pathname";
    self.datafolderPath = Path(self.datafolder);
    self.masker = NiftiLabelsMasker(labels_img="PATHNAMEPLACEHOLDER/schaefer200MNI.nii.gz");
    self.pathToBOLDFile = "";
    self.pathToConfoundsFile = "";
    self.eventsListPath = ""
  def buildTimeSeries(self):
    try:
      confound = pd.read_csv(self.pathToConfoundsFile, sep="\t");
      selected_confounds = [
          'global_signal',
          'white_matter',
          'csf',
          'trans_x', 'trans_x_derivative1', 'trans_x_derivative1_power2', 'trans_x_power2',
          'trans_y', 'trans_y_derivative1', 'trans_y_power2', 'trans_y_derivative1_power2',
          'trans_z', 'trans_z_derivative1', 'trans_z_derivative1_power2', 'trans_z_power2',
          'rot_x', 'rot_x_derivative1', 'rot_x_power2', 'rot_x_derivative1_power2',
          'rot_y', 'rot_y_derivative1', 'rot_y_power2', 'rot_y_derivative1_power2',
          'rot_z', 'rot_z_derivative1', 'rot_z_power2', 'rot_z_derivative1_power2'
      ];

      cleanedConfound = confound[selected_confounds].fillna(0);
      return self.masker.fit_transform(self.pathToBOLDFile, confounds = cleanedConfound);
    except Exception as error:
      raise RuntimeError("Check Pathname Hardcodes") from error;

  def splitConditions(self):
    timeSeries = self.buildTimeSeries();
    TSVData = pd.read_csv(self.eventsListPath,sep="\t");
    CTR = [[] for _ in range(0,7)];
    CTRF = [[] for _ in range(0,7)];
    self.conditions = ["Neutral - OBSERVAR", "Negativo - OBSERVAR", "Happy - OBSERVAR", "Negativo - REDUCIR", "Negativo - SUPRIMIR", "Happy - SUPRIMIR", "Happy - INCREMENTAR"]
    for row in TSVData.itertuples():
      for i in range(0,len(self.conditions)):
        if row[3] == self.conditions[i]:
          CTR[i].append(timeSeries[row[1]//2-1:(row[1]+row[2])//2,:]);
    for condition in range(0,len(CTR)):
      CTRF[condition] = pd.DataFrame(np.concatenate(CTR[condition]));
    self.saveTimeSeries = [df.to_numpy() for df in CTRF]
    return CTRF;

#this method needs to use Ledoit-wolf and proper fisher z shifiting
  def buildFCMatrices(self):
      CTR=self.splitConditions()
      CFCM=[[] for _ in range(7)]
      # ledoit-wolf 
      lw_measure=ConnectivityMeasure(kind='correlation', cov_estimator=LedoitWolf())
      for i in range(len(CTR)):
        array=CTR[i].to_numpy()
        r_matrix=lw_measure.fit_transform([array])[0]
        z_matrix=np.arctanh(r_matrix)
        np.fill_diagonal(z_matrix, 0)
        z_matrix[np.isinf(z_matrix)]=0  
        CFCM[i]=z_matrix    
        return CFCM
      
  #this is a new mehtod that converts the matrix objects into pytorch geo graph topologies
  def build_sparse(self,fc_matrix,top_k):
    num_nodes=fc_matrix.shape[0]
    sources=[]
    targets=[]
    edge_w=[]
    for node_idx in range(num_nodes):
      row=fc_matrix[node_idx]
      top_indices=np.argsort(np.abs(row))[-top_k:]
      for target_idx in top_indices:
        sources.append(node_idx)
        targets.append(target_idx)
        edge_w.append(row[target_idx])
      edge_index=torch.tensor([sources, targets], dtype=torch.long)
      edge_attr=torch.tensor(edge_w, dtype=torch.float).unsqueeze(-1)
      x=torch.ones((num_nodes, 5), dtype=torch.float) 
      return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
   
    # method iterate and arrange by iterpolated adge density
    def execute(self):
      try:
        top_k_list=[20, 30, 40, 50] 
        for subfolder in self.datafolderPath.iterdir():
          if subfolder.is_dir() and not subfolder.name.startswith("top_"):
            self.pathToBOLDFile=os.path.join(self.datafolder, subfolder.name, f"{subfolder.name}_BOLD.nii.gz")
            self.pathToConfoundsFile=os.path.join(self.datafolder, subfolder.name, f"{subfolder.name}_Confounds.tsv")
            self.eventsListPath=os.path.join(self.datafolder, subfolder.name, f"{subfolder.name}_events.tsv")
            c_matrices=self.buildFCMatrices()
            for top_k in top_k_list:
              output_dir=os.path.join(self.datafolder, f"top_{top_k}", subfolder.name)
              os.makedirs(output_dir, exist_ok=True)
              for i in range(len(c_matrices)):
                cond_name=self.conditions[i].replace(" ", "").replace("-", "")
                np.save(os.path.join(output_dir, f"{subfolder.name}_FCMatrix_Cond_{cond_name}.npy"), c_matrices[i])
                np.save(os.path.join(output_dir, f"{subfolder.name}_ROITimeSeries_Cond_{cond_name}.npy"), self.saveTimeSeries[i])
                graph_data=self.buildSparseGraph(c_matrices[i], top_k)
                torch.save(graph_data, os.path.join(output_dir, f"{subfolder.name}_PyGGraph_Cond_{cond_name}.pt"))              
      except Exception as error:
        raise RuntimeError("Check Pathname/Pipeline Parameters") from error  
      