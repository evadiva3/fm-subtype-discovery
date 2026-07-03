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
    self.saveTimeSeries = CTRF.to_numpy();
    return CTRF;
  def buildFCMatrices(self):
    CTR = self.splitConditions();
    CFCM = [[]for _ in range(0,7)];
    for i in range (0,len(CTR)):
      CFCM[i] = CTR[i].corr(method='pearson');
      CFCM[i] = np.arctanh(CFCM[i]);
      CFCM[i] = CFCM[i].replace([np.inf,-np.inf], 0);
    return CFCM;
  def execute(self):
    try:
      for subfolder in self.datafolderPath.iterdir():
        if subfolder.is_dir():
          self.pathToBOLDFile = self.datafolder + "/" + subfolder.name + "/" + subfolder.name + "_BOLD.nii.gz";
          self.pathToConfoundsFile = self.datafolder + "/" + subfolder.name + "/" + subfolder.name + "_Confounds.tsv";
          self.eventsListPath = self.datafolder + "/" + subfolder.name + "/" + subfolder.name + "_events.tsv";
          conditions = self.buildFCMatrices()
          for i in range(0,len(conditions)):
            np.save(self.datafolder + "/" + subfolder.name + "/" + subfolder.name + "_FCMatrixCondition" + self.conditions[i].replace(" ", "") + ".npy",conditions[i]);
            np.save(self.datafolder + "/" + subfolder.name + "/" + subfolder.name + "_ROITimeSeries" + self.conditions[i].replace(" ", "") + ".npy",self.saveTimeSeries[i]);
    except Exception as error:
      raise RuntimeError("Check Pathname Hardcodes") from error;