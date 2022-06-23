void particle_id_micro()
{
    TMVA::Tools::Instance();

    bool useFischer = true;       // Fischer discriminant
    bool useMLP = true;          // Multi Layer Perceptron (old TMVA NN implementation)
    bool useBDT = true;           // Boosted Decision Tree
    bool useKNN = true;
    bool useSVM = true;

    TString outfileName( "Micro_output.root" );
    auto outputFile = TFile::Open(outfileName, "RECREATE");

    TMVA::Factory factory("PI2", outputFile,
                          "!V:ROC:!Silent:Color:AnalysisType=Classification" );

    TString inputFileName = "MicroBooNE.root";
    TFile *inputFile = nullptr;

    inputFile = TFile::Open(inputFileName);

    TTree *signalTree     = (TTree*)inputFile->Get("signal");
    TTree *backgroundTree = (TTree*)inputFile->Get("background");

    signalTree->Print();

    TMVA::DataLoader *dataloader = new TMVA::DataLoader("dataset_micro");

    dataloader->AddVariable("var1", 'F');
    dataloader->AddVariable("var2", 'F');
    dataloader->AddVariable("var3", 'F' );
    dataloader->AddVariable("var4", 'F' );
    dataloader->AddVariable("var5", 'F' );
    dataloader->AddVariable("var6", 'F' );
    dataloader->AddVariable("var7", 'F' );
    dataloader->AddVariable("var8", 'F' );
    dataloader->AddVariable("var9", 'F' );
    dataloader->AddVariable("var10", 'F' );
    dataloader->AddVariable("var11", 'F' );
    dataloader->AddVariable("var12", 'F' );
    dataloader->AddVariable("var13", 'F' );
    dataloader->AddVariable("var14", 'F' );
    dataloader->AddVariable("var15", 'F' );
    dataloader->AddVariable("var16", 'F' );
    dataloader->AddVariable("var17", 'F' );
    dataloader->AddVariable("var18", 'F' );
    dataloader->AddVariable("var19", 'F' );
    dataloader->AddVariable("var20", 'F' );
    dataloader->AddVariable("var21", 'F' );
    dataloader->AddVariable("var22", 'F' );
    dataloader->AddVariable("var23", 'F' );
    dataloader->AddVariable("var24", 'F' );
    dataloader->AddVariable("var25", 'F' );
    dataloader->AddVariable("var26", 'F' );
    dataloader->AddVariable("var27", 'F' );
    dataloader->AddVariable("var28", 'F' );
    dataloader->AddVariable("var29", 'F' );
    dataloader->AddVariable("var30", 'F' );
    dataloader->AddVariable("var31", 'F' );
    dataloader->AddVariable("var32", 'F' );
    dataloader->AddVariable("var33", 'F' );
    dataloader->AddVariable("var34", 'F' );
    dataloader->AddVariable("var35", 'F' );
    dataloader->AddVariable("var36", 'F' );
    dataloader->AddVariable("var37", 'F' );
    dataloader->AddVariable("var38", 'F' );
    dataloader->AddVariable("var39", 'F' );
    dataloader->AddVariable("var40", 'F' );
    dataloader->AddVariable("var41", 'F' );
    dataloader->AddVariable("var42", 'F' );
    dataloader->AddVariable("var43", 'F' );
    dataloader->AddVariable("var44", 'F' );
    dataloader->AddVariable("var45", 'F' );
    dataloader->AddVariable("var46", 'F' );
    dataloader->AddVariable("var47", 'F' );
    dataloader->AddVariable("var48", 'F' );
    dataloader->AddVariable("var49", 'F' );
    dataloader->AddVariable("var50", 'F' );

    Double_t signalWeight     = 1.0;
    Double_t backgroundWeight = 1.0;

    dataloader->AddSignalTree(signalTree, signalWeight);
    dataloader->AddBackgroundTree(backgroundTree, backgroundWeight);

    TCut cuts = "";
    TCut cutb = "";

    dataloader->PrepareTrainingAndTestTree(cuts, cutb, "nTrain_Signal=7500:nTrain_Background=7500:nTest_Signal=2500:nTest_Background=2500:SplitMode=Random:!V" );

    // Fisher discriminant (same as LD)
    if (useFischer)
    {
        factory.BookMethod(dataloader, TMVA::Types::kFisher, "Fisher", "H:!V:Fisher:VarTransform=G:CreateMVAPdfs:PDFInterpolMVAPdf=Spline2:NbinsMVAPdf=50:NsmoothMVAPdf=10" );
    }

    //Boosted Decision Trees
    if (useBDT)
    {
        factory.BookMethod(dataloader,TMVA::Types::kBDT, "BDT",
                           "!V:NTrees=400:MinNodeSize=2.5%:MaxDepth=3:BoostType=AdaBoost:AdaBoostBeta=0.2:UseBaggedBoost:BaggedSampleFraction=0.5:SeparationType=GiniIndex:nCuts=20" );
    }

    //Multi-Layer Perceptron (Neural Network)
    if (useMLP)
    {
        factory.BookMethod(dataloader, TMVA::Types::kMLP, "MLP",
                           "!H:!V:NeuronType=tanh:VarTransform=N:NCycles=100:HiddenLayers=N+5:TestRate=5:!UseRegulator" );
    }

    if (useKNN)
    {
        factory.BookMethod( dataloader, TMVA::Types::kKNN, "KNN",
                            "H:nkNN=20:ScaleFrac=0.8:SigmaFact=1.0:Kernel=Gaus:UseKernel=F:UseWeight=T:!Trim" );
    }

    if (useSVM)
    {

        factory.BookMethod( dataloader, TMVA::Types::kSVM, "SVM", "Gamma=0.25:Tol=0.001:VarTransform=Norm" );

    }

    factory.TrainAllMethods();
    factory.TestAllMethods();
    factory.EvaluateAllMethods();

    auto c1 = factory.GetROCCurve(dataloader);
    c1->Draw();

    outputFile->Close();

    if (!gROOT->IsBatch()) TMVA::TMVAGui(outfileName);
}

