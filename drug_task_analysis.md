## Dataset Analysis

### Experimental Design & Scale

- The sci-Plex study applied sci-RNA-seq2 with nuclear hashing to screen **188 small-molecule compounds** across **four doses** (10 nM–10 µM) plus vehicle controls, profiling **∼650 000** cells across **∼5 000** independent conditions in A549, K562, and MCF7 lines 
    
- Each cell yields UMI counts for **~20 000** genes (median ~1 000–2 400 UMIs), with **>90 % zeros** per cell reflecting dropout and lowly expressed transcripts 

### Experimental Setup

- **Biological Conditions**: Three human cancer cell lines (A549 lung adenocarcinoma, K562 chronic myeloid leukemia, MCF7 breast adenocarcinoma) representing diverse tissue contexts and baseline transcriptional states
    
- **Experimental Protocol**: sci-RNA-seq2 with nuclear hashing enables multiplexed single-cell profiling across thousands of independent perturbation conditions in a single experiment
    
- **Controls**: Vehicle controls (DMSO) for each cell line and dose level provide baseline expression references
    
- **Replicates**: High concordance (r≈0.99) between biological replicates demonstrates experimental reproducibility
    
- **Biological Validation**: Replicate concordance and dose-response relationships validate the experimental design and data quality

### Perturbation Design

- **Perturbation Types**: Small-molecule compounds targeting diverse cellular pathways (kinase inhibitors, epigenetic modulators, metabolic inhibitors, etc.)
    
- **Dose Range**: Four log-spaced doses (10 nM, 100 nM, 1 µM, 10 µM) plus vehicle control enable dose-response modeling
    
- **Timepoints**: Single timepoint measurement (typically 24-48 hours post-treatment) captures steady-state transcriptional responses
    
- **Biological Controls**: Vehicle controls and untreated cells serve as negative controls; known pathway activators serve as positive controls

### Sample Characteristics

- **Sample Size**: ~650,000 cells distributed across ~5,000 independent conditions (188 compounds × 4 doses × 3 cell lines + controls)
    
- **Distribution**: Uneven distribution across conditions reflects experimental design priorities and compound availability
    
- **Characteristics**: Each condition typically contains 50-200 cells, sufficient for robust statistical analysis
    
- **Biological Variability**: Intrinsic cell-to-cell variation reflects cell cycle, pathway activity, and stochastic gene expression
    
- **Technical Variability**: Hash collision noise and plate effects contribute to technical variation requiring normalization

### Data Characteristics

- **Sparsity & Noise**: Single‐cell RNA–seq dropouts dominate, compounded by "hash collision" noise in demultiplexing 
    
- **Heterogeneity**: Intrinsic variability (cell cycle, pathway activity) and cell‐line–specific baselines require models to condition on baseline expression
    
- **Batch Effects**: Though replicates show high concordance (r≈0.99), subtle plate‐ and hash‐driven batch artifacts persist, necessitating robust normalization and potential covariate modeling

### Data Properties

#### Distribution Patterns

- **Biological Patterns**: 
  - Cell-line-specific expression signatures reflect tissue-of-origin and cancer type
  - Dose-dependent transcriptional shifts follow expected pharmacological dose-response relationships
  - Pathway-specific responses align with known drug mechanisms of action
  
- **Technical Patterns**:
  - Zero-inflated count distributions typical of single-cell RNA-seq
  - Log-normal distribution of non-zero expression values
  - Plate and hash batch effects detectable via PCA and correlation analysis

#### Feature Space

- **Biological Features**: 
  - ~20,000 genes covering the full transcriptome
  - Highly variable genes (HVGs) capture cell-type and perturbation-specific responses
  - Pathway-enriched gene sets enable biological interpretation
  
- **Technical Features**:
  - UMI counts provide quantitative expression measurements
  - Hash barcodes enable multiplexed sample identification
  - Quality metrics (hash_umis, qval_demultiplexing) enable quality control

#### Quality Metrics

- **Biological Quality**:
  - High replicate concordance (r≈0.99) validates biological reproducibility
  - Dose-response relationships demonstrate expected pharmacological behavior
  - Pathway enrichment validates biological relevance
  
- **Technical Quality**:
  - Median 1,000-2,400 UMIs per cell indicates good library complexity
  - >90% zeros per cell reflects typical single-cell dropout rates
  - Hash collision rates <5% enable reliable demultiplexing

#### Noise Sources

- **Biological Noise**:
  - Cell-to-cell transcriptional variability (intrinsic biological noise)
  - Cell cycle and pathway activity heterogeneity
  - Stochastic gene expression
  
- **Technical Noise**:
  - Single-cell RNA-seq dropout events
  - Hash collision misassignment
  - Plate and batch effects

### Preprocessing Implications

- **Gene Filtering**: Remove genes detected in < 1 % of cells to reduce noise and dimensionality.
    
- **Normalization**: Log‐CP10K (`log2(CP10K+1)`) followed by z-scoring per gene across training cells to harmonize distributions across lines.
    
- **Feature Selection**: Focus on the top 2 000–5 000 highly variable genes to balance expressivity and tractability.
    
- **Perturbation Encoding**: Learn embeddings for each compound ID and normalize log₁₀-dose to [0, 1].

### Preprocessing Recommendations

#### Normalization Strategies

- **Biological Normalization**:
  - Log-CP10K normalization: `log2(CP10K+1)` transforms count data to log space while preserving zeros
  - Per-gene z-scoring across training cells harmonizes expression distributions across cell lines
  - Rationale: Log transformation stabilizes variance; z-scoring enables cross-line comparison
  
- **Technical Normalization**:
  - Batch correction using Harmony or scVI to remove plate and hash effects
  - Quality control filtering based on hash_umis and demultiplexing quality scores
  - Implementation: Standard scanpy/scvi-tools pipelines

#### Quality Control Procedures

- **Biological QC**:
  - Filter cells with <200 detected genes (low-quality cells)
  - Remove cells with >10% mitochondrial gene expression (dying cells)
  - Validate dose-response relationships for known compounds
  
- **Technical QC**:
  - Filter cells with hash_umis <100 (poor demultiplexing)
  - Remove cells with qval_demultiplexing >0.1 (ambiguous assignment)
  - Filter genes detected in <1% of cells (lowly expressed)

#### Feature Selection Methods

- **Biological Feature Selection**:
  - Highly variable gene (HVG) selection using scanpy's `sc.pp.highly_variable_genes`
  - Pathway-enriched gene sets for biological interpretation
  - Top 2,000-5,000 HVGs balance expressivity and computational tractability
  
- **Technical Feature Selection**:
  - Remove ribosomal and mitochondrial genes if not biologically relevant
  - Retain genes with consistent detection across conditions
  - Criteria: Coefficient of variation >0.5, mean expression >0.1

#### Batch Correction Approaches

- **Biological Batch Correction**:
  - Harmony integration to remove plate effects while preserving biological variation
  - scVI batch correction for probabilistic batch effect removal
  - Rationale: Preserve cell-line and perturbation effects while removing technical artifacts
  
- **Technical Batch Correction**:
  - Hash batch correction using linear models
  - Plate effect correction using ComBat or similar methods
  - Implementation: Standard batch correction pipelines

### Quality Assessment

#### Completeness

- **Coverage**: 
  - ~5,000 independent conditions provide comprehensive coverage of compound × dose × cell-line space
  - Some compounds may have missing doses or cell lines due to experimental constraints
  - Missing data patterns should be analyzed for potential biases
  
- **Missing Data**: 
  - Some compound-cell-line combinations may be missing
  - Missing doses for certain compounds
  - Impact: May limit generalization to unseen compound-cell-line combinations
  - Mitigation: Focus on well-covered conditions for training; use transfer learning for sparse conditions

#### Relevance

- **Biological Significance**: 
  - Drug perturbation prediction directly addresses pharmaceutical research needs
  - Three cancer cell lines represent diverse tissue contexts
  - 188 compounds cover diverse mechanisms of action
  
- **Technical Significance**: 
  - Large-scale single-cell perturbation data enables robust model training
  - Multiplexed experimental design enables efficient data collection
  - High-quality data enables reliable model evaluation

#### Reproducibility

- **Technical Reproducibility**: 
  - High replicate concordance (r≈0.99) demonstrates technical reproducibility
  - Standardized experimental protocols ensure consistency
  - Quality metrics enable reproducibility assessment
  
- **Biological Reproducibility**: 
  - Dose-response relationships validate biological reproducibility
  - Pathway enrichment validates expected biological responses
  - Cross-validation enables reproducibility assessment

#### Limitations

- **Biological Limitations**:
  - Single timepoint may miss dynamic responses
  - Three cell lines may not capture full tissue diversity
  - In vitro conditions may not fully recapitulate in vivo responses
  
- **Technical Limitations**:
  - Hash collision noise may affect demultiplexing accuracy
  - Batch effects may confound biological signals
  - Dropout rates may limit detection of lowly expressed genes
    

## Task Analysis

### Problem Definition

#### Formal Definition

- **Biological Context**: Predicting how small-molecule drug perturbations alter gene expression in human cancer cell lines, enabling in silico screening of compound effects and dose-response relationships
    
- **Input-Output Mapping**: 
  - **Input 1**: Baseline (control) expression vector x_baseline ∈ ℝ^d for a cell from one of three cell lines (A549, K562, MCF7), where d ≈ 2,000-20,000 genes
  - **Input 2**: Perturbation descriptor p = (compound_id, dose), where compound_id is categorical (one of 188 compounds) and dose is continuous (log₁₀-transformed molarity, typically 10 nM to 10 µM)
  - **Output**: Predicted post-perturbation expression vector ŷ ∈ ℝ^d representing the expected transcriptional shift induced by the drug
    
- **Biological Constraints**: 
  - Models must respect dose-response relationships (monotonic or sigmoidal dose effects)
  - Predictions should preserve cell-line-specific baseline expression patterns
  - Pathway-level predictions should align with known drug mechanisms of action
  
- **Evaluation Criteria**: 
  - Mean Squared Error (MSE) between predicted and observed expression across all genes
  - Pearson Correlation Coefficient (PCC) over all genes (computed as Pearson correlation between flattened prediction and target matrices)
  - R² (Coefficient of Determination) computed over all genes
  - MSE_DE, PCC_DE, R²_DE computed over top-k differentially expressed genes per perturbation (k=20, selected based on absolute log2 fold change relative to control)
  - Additional biological validation metrics (pathway enrichment, dose-response curves) for comprehensive assessment
  
- **Success Metrics**: 
  - MSE < 0.5 (normalized expression space)
  - Global PCC > 0.7
  - PCC_DE > 0.8 (computed over top-20 DE genes per perturbation)
  - R² > 0.5 (indicates model explains >50% of variance)

#### Key Variables

- **Biological Variables**: 
  - Baseline gene expression (cell state)
  - Cell line identity (tissue context)
  - Compound identity (mechanism of action)
  - Dose level (pharmacological intensity)
  - Post-perturbation expression (response)
  
- **Technical Variables**: 
  - Normalized expression values (log-CP10K, z-scored)
  - Perturbation encoding (one-hot or learned embeddings)
  - Dose encoding (log₁₀-transformed, normalized to [0,1])
  - Batch and plate identifiers
  
- **Relationships**: 
  - Baseline expression modulates drug response (cell-state-dependent effects)
  - Dose determines response magnitude (dose-response relationships)
  - Compound identity determines response pattern (mechanism-specific effects)
  - Cell line identity affects baseline and response (tissue-specific effects)
  
- **Constraints**: 
  - Expression values must be non-negative (count data)
  - Dose-response relationships should be monotonic or sigmoidal
  - Predictions should preserve biological pathway structure
  
- **Validation Requirements**: 
  - Held-out compound evaluation (unseen drugs)
  - Held-out cell line evaluation (unseen tissues)
  - Dose-response curve validation
  - Pathway enrichment validation

#### Scope

- **Biological Scope**: 
  - Three human cancer cell lines (A549, K562, MCF7) representing diverse tissue contexts
  - 188 small-molecule compounds covering diverse mechanisms of action
  - Single timepoint measurement (steady-state responses)
  - Focus on transcriptional responses (RNA expression)
  
- **Technical Scope**: 
  - Supervised learning from paired baseline-perturbation observations
  - Regression task (continuous expression prediction)
  - High-dimensional output space (2,000-20,000 genes)
  - Sparse, zero-inflated data
  
- **Limitations**: 
  - Single timepoint may miss dynamic responses
  - In vitro conditions may not fully recapitulate in vivo responses
  - Limited to three cell lines (may not generalize to all tissues)
  - Focus on transcriptional responses (may miss post-transcriptional effects)
  
- **Assumptions**: 
  - Baseline expression captures relevant cell state information
  - Dose-response relationships are learnable from data
  - Compound effects are consistent across similar cell states
  - Expression changes reflect direct and indirect drug effects
  
- **Biological Validation**: 
  - Dose-response relationships should match expected pharmacological behavior
  - Pathway enrichment should align with known drug mechanisms
  - Predictions should generalize to unseen compounds and cell lines

### Key Challenges

#### Biological Challenges

- **Challenges**: 
  - Cell-to-cell heterogeneity in baseline states and responses
  - Cell-line-specific baseline expression patterns
  - Diverse drug mechanisms of action requiring flexible representations
  - Dose-response relationships that may be non-linear or cell-state-dependent
  
- **Impact**: 
  - Models must learn cell-state-dependent drug responses
  - Baseline conditioning is essential for accurate predictions
  - Compound representations must capture mechanism-specific effects
  - Dose encoding must enable interpolation and extrapolation
  
- **Mitigation**: 
  - Use baseline expression as conditional input
  - Learn compound embeddings that capture mechanism-specific patterns
  - Model dose effects explicitly (e.g., dose-response modules)
  - Use attention mechanisms to focus on relevant genes and pathways
  
- **Validation**: 
  - Evaluate on held-out compounds to test mechanism generalization
  - Evaluate on held-out cell lines to test tissue generalization
  - Validate dose-response relationships for known compounds
  - Compare pathway enrichment with known drug mechanisms
  
- **Biological Considerations**: 
  - Models should respect biological constraints (e.g., pathway structure)
  - Predictions should be interpretable (e.g., pathway-level effects)
  - Generalization should reflect biological principles (e.g., similar compounds have similar effects)

#### Technical Challenges

- **Challenges**: 
  - High-dimensional output space (2,000-20,000 genes) requires efficient architectures
  - Extreme sparsity (>90% zeros) requires robust loss functions
  - Batch effects may confound biological signals
  - Computational scalability for large datasets (~650,000 cells)
  
- **Impact**: 
  - Standard architectures may struggle with high-dimensional outputs
  - MSE loss may be dominated by zeros
  - Batch effects may reduce generalization
  - Training on large datasets requires efficient implementations
  
- **Mitigation**: 
  - Use dimensionality reduction or feature selection (HVGs)
  - Implement zero-inflated or negative binomial loss functions
  - Apply batch correction or include batch as covariate
  - Use efficient architectures (e.g., transformers, graph neural networks)
  
- **Validation**: 
  - Compare performance with and without batch correction
  - Evaluate computational efficiency (training time, memory usage)
  - Test robustness to different preprocessing choices
  - Validate scalability to larger datasets
  
- **Implementation Requirements**: 
  - Efficient data loading and batching
  - GPU acceleration for large-scale training
  - Reproducible preprocessing pipelines
  - Modular architecture design for experimentation

#### Data Quality Challenges

- **Issues**: 
  - Single-cell RNA-seq dropout events (>90% zeros)
  - Hash collision noise in demultiplexing
  - Plate and batch effects
  - Missing conditions (some compound-cell-line combinations)
  
- **Impact**: 
  - Dropout may limit detection of lowly expressed genes
  - Hash collisions may introduce misassignment errors
  - Batch effects may confound biological signals
  - Missing conditions may limit training data
  
- **Mitigation**: 
  - Use zero-inflated loss functions to handle dropout
  - Filter cells with poor demultiplexing quality
  - Apply batch correction methods
  - Use transfer learning or data augmentation for sparse conditions
  
- **Validation**: 
  - Evaluate performance with different quality control thresholds
  - Compare with and without batch correction
  - Test robustness to missing data
  - Validate predictions on high-quality subsets
  
- **Biological Validation**: 
  - Validate predictions on high-quality replicates
  - Compare pathway enrichment with and without quality control
  - Test generalization to different quality regimes

#### Computational Challenges

- **Challenges**: 
  - Training on ~650,000 cells requires efficient implementations
  - High-dimensional outputs require memory-efficient architectures
  - Hyperparameter optimization for complex models
  - Reproducibility across different hardware and software environments
  
- **Impact**: 
  - Training time may be prohibitive without optimization
  - Memory constraints may limit model complexity
  - Hyperparameter search may be computationally expensive
  - Reproducibility may be affected by non-deterministic operations
  
- **Mitigation**: 
  - Use efficient data loading (e.g., HDF5, memory mapping)
  - Implement gradient checkpointing for memory efficiency
  - Use automated hyperparameter optimization (e.g., Optuna)
  - Set random seeds and use deterministic operations
  
- **Validation**: 
  - Benchmark training time and memory usage
  - Compare hyperparameter optimization strategies
  - Test reproducibility across different environments
  - Validate computational efficiency improvements
  
- **Resource Requirements**: 
  - GPU memory: 16-32 GB for large models
  - Training time: 1-7 days depending on model complexity
  - Storage: 100-500 GB for datasets and model checkpoints
  - CPU: Multi-core for data preprocessing

#### Interpretability Challenges

- **Requirements**: 
  - Identify key genes and pathways driving predictions
  - Understand dose-response relationships
  - Interpret compound-specific effects
  - Explain cell-line-specific differences
  
- **Challenges**: 
  - High-dimensional outputs make interpretation difficult
  - Black-box models may lack interpretability
  - Pathway-level interpretation requires gene set analysis
  - Dose-response interpretation requires visualization
  
- **Solutions**: 
  - Use attention mechanisms to identify important genes
  - Implement pathway enrichment analysis
  - Visualize dose-response curves for key genes
  - Use model interpretability methods (e.g., SHAP, gradient-based attribution)
  
- **Validation**: 
  - Compare attention weights with known pathway annotations
  - Validate pathway enrichment with known drug mechanisms
  - Test interpretability on held-out compounds
  - Evaluate user understanding of model predictions
  
- **Biological Validation**: 
  - Interpretations should align with known biology
  - Pathway-level effects should match expected mechanisms
  - Dose-response relationships should be biologically plausible
  - Cell-line-specific effects should reflect tissue context

### Research Questions

#### Primary Research Questions

- **Questions**: 
  - Can deep learning models accurately predict drug-induced transcriptional changes from baseline expression and perturbation descriptors?
  - How do different model architectures compare in their ability to capture dose-response relationships and generalize to unseen compounds?
  - What are the key biological and technical factors that determine prediction accuracy?
  - Can models learn interpretable representations of drug mechanisms of action?
  
- **Hypotheses**: 
  - Models that condition on baseline expression will outperform those that do not
  - Explicit dose encoding will improve dose-response prediction
  - Attention mechanisms will improve interpretability and performance
  - Transfer learning from related compounds will improve generalization
  
- **Biological Significance**: 
  - Accurate prediction enables in silico drug screening
  - Understanding dose-response relationships informs drug dosing
  - Interpretable models enable mechanism discovery
  - Generalization to unseen compounds enables drug repurposing
  
- **Validation Approach**: 
  - Evaluate on held-out compounds and cell lines
  - Compare with baseline methods (CPA, cycleCDR, etc.)
  - Validate dose-response relationships and pathway enrichment
  - Test interpretability through attention analysis and pathway enrichment
  
- **Expected Outcomes**: 
  - Models that achieve MSE < 0.5, PCC > 0.7, and R² > 0.5 (global metrics)
  - Models that achieve PCC_DE > 0.8 and R²_DE > 0.6 (DE-specific metrics)
  - Improved generalization to unseen compounds compared to baselines
  - Interpretable predictions that align with known biology
  - Computational frameworks for large-scale drug perturbation prediction

#### Secondary Research Questions

- **Questions**: 
  - How do different preprocessing strategies affect model performance?
  - What is the optimal number of highly variable genes for prediction?
  - How do batch effects impact generalization?
  - Can models predict cell-line-specific responses?
  
- **Hypotheses**: 
  - Log-CP10K normalization with z-scoring will outperform alternatives
  - 2,000-5,000 HVGs will balance expressivity and tractability
  - Batch correction will improve generalization
  - Cell-line-specific models will outperform universal models
  
- **Biological Significance**: 
  - Preprocessing choices affect biological signal preservation
  - Feature selection balances biological relevance and computational efficiency
  - Batch correction enables cross-experiment generalization
  - Cell-line-specific models capture tissue context
  
- **Validation Approach**: 
  - Compare preprocessing strategies via cross-validation
  - Test different HVG numbers
  - Evaluate with and without batch correction
  - Compare universal vs. cell-line-specific models
  
- **Expected Outcomes**: 
  - Optimal preprocessing pipeline for drug perturbation prediction
  - Recommended HVG numbers for different model architectures
  - Batch correction strategies that preserve biological signals
  - Understanding of when cell-line-specific models are beneficial

#### Biological Mechanisms

- **Mechanisms**: 
  - Direct target binding and pathway activation
  - Indirect effects through regulatory networks
  - Dose-dependent response modulation
  - Cell-state-dependent response variation
  
- **Investigation Approach**: 
  - Pathway enrichment analysis of predicted changes
  - Attention weight analysis to identify key genes
  - Dose-response curve analysis
  - Cell-line-specific effect analysis
  
- **Validation Methods**: 
  - Compare pathway enrichment with known drug mechanisms
  - Validate attention weights with pathway annotations
  - Test dose-response relationships for known compounds
  - Evaluate cell-line-specific effects against tissue context
  
- **Expected Insights**: 
  - Understanding of how drugs alter gene expression
  - Identification of key pathways and genes
  - Dose-response mechanisms
  - Cell-state-dependent response mechanisms
  
- **Biological Validation**: 
  - Mechanisms should align with known pharmacology
  - Pathway effects should match expected drug actions
  - Dose-response relationships should be biologically plausible
  - Cell-line-specific effects should reflect tissue biology

#### Experimental Validation

- **Requirements**: 
  - Independent validation on held-out compounds
  - Cross-validation within training compounds
  - Dose-response validation for known compounds
  - Pathway enrichment validation
  
- **Methods**: 
  - Held-out compound evaluation (unseen drugs)
  - Held-out cell line evaluation (unseen tissues)
  - Dose-response curve analysis
  - Pathway enrichment analysis (GSEA, GSVA)
  
- **Controls**: 
  - Baseline models (CPA, cycleCDR, etc.)
  - Random predictions
  - Baseline expression (no change prediction)
  
- **Metrics**: 
  - Primary metrics: MSE, PCC, R² (computed over all genes)
  - DE-specific metrics: MSE_DE, PCC_DE, R²_DE (computed over top-20 DE genes per perturbation, selected based on absolute log2 fold change)
  - Additional biological metrics: Pathway enrichment correlation (for biological validation, not part of core evaluation)
  - Dose-response curve accuracy
  - Cell-line-specific performance
  
- **Biological Validation**: 
  - Predictions should match experimental observations
  - Pathway enrichment should align with known mechanisms
  - Dose-response relationships should be pharmacologically plausible
  - Generalization should reflect biological principles

### Analysis Methods

#### Computational Approaches

- **Methods**: 
  - Deep learning architectures (transformers, graph neural networks, VAEs)
  - Baseline conditioning via concatenation or attention
  - Compound embedding learning
  - Dose-response modeling (explicit dose encoding)
  
- **Rationale**: 
  - Deep learning enables flexible non-linear modeling
  - Baseline conditioning captures cell-state-dependent effects
  - Compound embeddings learn mechanism-specific representations
  - Explicit dose encoding enables dose-response prediction
  
- **Implementation**: 
  - PyTorch-based model implementations
  - Standardized data loading and preprocessing pipelines
  - Modular architecture design for experimentation
  - Reproducible training and evaluation scripts
  
- **Validation**: 
  - Compare with baseline methods
  - Ablation studies on architecture components
  - Hyperparameter optimization
  - Cross-validation on training data
  
- **Biological Validation**: 
  - Pathway enrichment analysis
  - Attention weight interpretation
  - Dose-response curve validation
  - Mechanism-specific effect analysis

#### Validation Strategies

- **Strategies**: 
  - Held-out compound evaluation (unseen drugs)
  - Held-out cell line evaluation (unseen tissues)
  - Cross-validation within training compounds
  - Dose-response validation
  
- **Rationale**: 
  - Held-out evaluation tests true generalization
  - Cross-validation enables robust performance estimation
  - Dose-response validation tests pharmacological plausibility
  - Multiple validation strategies ensure robustness
  
- **Implementation**: 
  - Standardized train-test splits
  - Cross-validation frameworks
  - Dose-response analysis pipelines
  - Pathway enrichment pipelines
  
- **Metrics**: 
  - Primary metrics: MSE, PCC, R² (computed over all genes)
  - DE-specific metrics: MSE_DE, PCC_DE, R²_DE (computed over top-20 DE genes per perturbation)
  - Additional biological metrics: Pathway enrichment correlation (for biological validation)
  - Dose-response curve accuracy
  - Cell-line-specific performance
  
- **Biological Validation**: 
  - Predictions should match experimental observations
  - Pathway enrichment should align with known mechanisms
  - Dose-response relationships should be pharmacologically plausible
  - Generalization should reflect biological principles

#### Analysis Pipelines

- **Pipelines**: 
  1. Data preprocessing (normalization, quality control, feature selection)
  2. Model training (architecture selection, hyperparameter optimization)
  3. Model evaluation (held-out evaluation, cross-validation)
  4. Biological interpretation (pathway enrichment, attention analysis)
  
- **Components**: 
  - Preprocessing: scanpy-based pipelines
  - Training: PyTorch-based training loops
  - Evaluation: Standardized evaluation scripts
  - Interpretation: Pathway enrichment and visualization tools
  
- **Workflow**: 
  - Preprocess data → Train models → Evaluate performance → Interpret results
  - Iterative refinement based on validation results
  - Ablation studies to understand model components
  - Biological validation of predictions
  
- **Validation**: 
  - Reproducibility across different environments
  - Consistency across different random seeds
  - Robustness to preprocessing choices
  - Biological plausibility of predictions
  
- **Biological Interpretation**: 
  - Pathway-level analysis of predicted changes
  - Gene-level attention weight analysis
  - Dose-response curve interpretation
  - Cell-line-specific effect interpretation

#### Experimental Validation

- **Methods**: 
  - Independent experimental validation (if available)
  - Literature-based validation of pathway effects
  - Comparison with known drug mechanisms
  - Dose-response validation for known compounds
  
- **Controls**: 
  - Baseline models (CPA, cycleCDR, etc.)
  - Random predictions
  - Baseline expression (no change prediction)
  
- **Metrics**: 
  - Experimental agreement (if available)
  - Literature concordance
  - Mechanism alignment
  - Dose-response accuracy
  
- **Analysis**: 
  - Statistical comparison with baselines
  - Pathway enrichment comparison
  - Dose-response curve comparison
  - Mechanism-specific effect comparison
  
- **Biological Validation**: 
  - Experimental validation confirms predictions
  - Literature concordance validates mechanisms
  - Mechanism alignment confirms biological relevance
  - Dose-response accuracy confirms pharmacological plausibility

#### Reproducibility

- **Standards**: 
  - Code versioning (Git)
  - Environment specification (conda/pip)
  - Random seed setting
  - Documentation of preprocessing steps
  
- **Requirements**: 
  - Reproducible data preprocessing
  - Reproducible model training
  - Reproducible evaluation
  - Reproducible interpretation
  
- **Validation**: 
  - Reproducibility across different environments
  - Consistency across different random seeds
  - Reproducibility of published results
  - Reproducibility of baseline comparisons
  
- **Documentation**: 
  - Detailed preprocessing documentation
  - Model architecture documentation
  - Training procedure documentation
  - Evaluation procedure documentation
  
- **Biological Validation**: 
  - Reproducible biological interpretations
  - Consistent pathway enrichment results
  - Reproducible dose-response relationships
  - Consistent mechanism-specific effects

### Evaluation Metrics

The model evaluation uses the following metrics, computed as implemented in the model code:

#### Primary Metrics (All Genes)

- **MSE (Mean Squared Error)**: 
  - **Calculation**: `MSE = mean((pred - true)²)` over all genes and all cells
  - **Purpose**: Measures average squared difference between predicted and observed expression
  - **Characteristics**: Sensitive to outliers and large errors
  - **Target**: MSE < 0.5 (normalized expression space)
  
- **PCC (Pearson Correlation Coefficient)**: 
  - **Calculation**: `PCC = pearsonr(true_flat, pred_flat)[0]` where `_flat` denotes flattened matrices
  - **Purpose**: Measures linear correlation between predicted and observed expression
  - **Characteristics**: Less sensitive to scale differences than MSE, ranges from -1 to 1
  - **Target**: Global PCC > 0.7
  
- **R² (Coefficient of Determination)**: 
  - **Calculation**: `R² = 1 - (SS_res / SS_tot)` where `SS_res = sum((true - pred)²)` and `SS_tot = sum((true - mean(true))²)`
  - **Purpose**: Measures the proportion of variance in observed expression explained by the predictions
  - **Characteristics**: Commonly used to assess goodness-of-fit in regression, ranges from -∞ to 1
  - **Target**: R² > 0.5 (indicates model explains >50% of variance)

#### DE-Specific Metrics (Differentially Expressed Genes)

- **MSE_DE**: 
  - **Calculation**: `MSE_DE = mean((pred_DE - true_DE)²)` computed over top-k DE genes per perturbation
  - **DE Gene Selection**: For each perturbation, select top-20 genes with highest absolute log2 fold change relative to control: `lfc = abs(log2((mean(true_pert) + ε) / (mean(control) + ε)))`
  - **Purpose**: Focuses evaluation on genes with largest perturbation-induced changes
  - **Target**: Lower MSE_DE indicates better prediction of key response genes
  
- **PCC_DE**: 
  - **Calculation**: `PCC_DE = pearsonr(true_DE_flat, pred_DE_flat)[0]` computed over top-20 DE genes per perturbation, then averaged across perturbations
  - **Purpose**: Measures correlation specifically for genes with largest changes
  - **Target**: PCC_DE > 0.8 (indicates strong correlation for key response genes)
  
- **R²_DE**: 
  - **Calculation**: `R²_DE = 1 - (SS_res_DE / SS_tot_DE)` computed over top-20 DE genes per perturbation, then averaged across perturbations
  - **Purpose**: Measures variance explained specifically for DE genes
  - **Target**: R²_DE > 0.6 (indicates model explains >60% of variance in key response genes)

#### Additional Biological Validation Metrics

- **Pathway Enrichment Correlation**: 
  - **Purpose**: Measures correlation between predicted and observed pathway enrichment scores (e.g., GSEA, GSVA)
  - **Status**: Additional biological validation metric, not part of core model evaluation
  - **Target**: Pathway enrichment correlation > 0.6 (when computed)
  
- **Dose-Response Curve Accuracy**: 
  - **Purpose**: Measures accuracy of predicted dose-response relationships for key drug-gene pairs
  - **Status**: Additional biological validation metric, not part of core model evaluation
  - **Target**: Dose-response correlation > 0.7 (when computed)
  
- **Cell-Line-Specific Performance**: 
  - **Purpose**: Measures performance separately for each cell line to validate tissue-specific generalization
  - **Status**: Can be computed by subsetting evaluation results by cell line
  - **Target**: Consistent performance across cell lines
    

## Baseline Models Analysis

### Literature Review

#### Existing Methods

Below is a survey of leading deep‐learning methods for single‐cell perturbation prediction, noting their strengths and limitations.

**Methods**: 
- Compositional Perturbation Autoencoder (CPA)
- cycleCDR
- CINEMA-OT
- ContrastiveVI
- scGen & Related Embedding Arithmetic

**Strengths**: 
- CPA: Captures compositionality (drugs + doses + time), strong interpolation to unseen dosages
- cycleCDR: Encourages preservation of baseline states, improves generalization to novel drugs
- CINEMA-OT: Theoretically grounded in causal inference, provides individual treatment-effect estimates
- ContrastiveVI: Improves separation of treatment effects from baseline variation, robust to technical noise
- scGen: Simplicity and interpretability of latent arithmetic, generalizes across cell types

**Limitations**: 
- CPA: Relies on adversarial disentanglement which can be unstable, struggles with extremely high-dimensional outputs
- cycleCDR: Cycle losses can collapse when perturbations induce large shifts, requires careful loss weighting
- CINEMA-OT: Scalability issues for large perturbation panels, requires good proxy for confounders
- ContrastiveVI: Doesn't explicitly model dose or continuous perturbation attributes, pretraining on control vs. treated pairs only
- scGen: Best suited to small perturbation sets, lacks explicit dose encoding

**Biological Relevance**: 
- CPA: Compositional design aligns with biological reality (drugs + doses + time)
- cycleCDR: Baseline preservation respects biological constraints
- CINEMA-OT: Causal inference framework enables biological interpretation
- ContrastiveVI: Separation of treatment effects enables mechanism discovery
- scGen: Latent arithmetic provides interpretable mechanism representations

#### Model Comparison

**Comparison Criteria**: 
- Prediction accuracy (MSE, PCC)
- Generalization to unseen compounds and cell lines
- Dose-response modeling capability
- Computational efficiency
- Interpretability

**Performance Metrics**: 
- MSE: Typically 0.3-0.8 (normalized expression space)
- PCC: Typically 0.6-0.8 (global, computed over all genes)
- PCC_DE: Typically 0.7-0.9 (computed over top-20 DE genes per perturbation)
- Dose-response accuracy: Varies by method
- Generalization: Varies by method and evaluation setup

**Biological Interpretability**: 
- CPA: Moderate (compositional embeddings provide some interpretability)
- cycleCDR: Low (cycle consistency doesn't directly provide interpretability)
- CINEMA-OT: High (causal inference framework enables interpretation)
- ContrastiveVI: Moderate (contrastive learning provides some interpretability)
- scGen: High (latent arithmetic is interpretable)

**Computational Requirements**: 
- CPA: Moderate (VAE training, adversarial training can be slow)
- cycleCDR: Moderate (cycle consistency requires forward-backward passes)
- CINEMA-OT: High (optimal transport is computationally expensive)
- ContrastiveVI: Moderate (contrastive learning requires negative sampling)
- scGen: Low (simple autoencoder architecture)

#### Recommendations

**Model Selection**: 
- For large-scale datasets (100+ compounds): CPA or cycleCDR
- For interpretability: CINEMA-OT or scGen
- For dose-response modeling: CPA (explicit dose encoding)
- For generalization: cycleCDR or ContrastiveVI

**Implementation Priority**: 
- High priority: CPA (compositionality, dose modeling)
- Medium priority: cycleCDR (generalization), ContrastiveVI (robustness)
- Low priority: CINEMA-OT (computational cost), scGen (scalability limitations)

**Biological Validation**: 
- All methods should be validated on held-out compounds and cell lines
- Dose-response relationships should be validated for known compounds
- Pathway enrichment should be validated against known drug mechanisms
- Interpretability should be validated through attention analysis or pathway enrichment

### Evaluation Framework

#### Metrics

**Primary Metrics**: 
- MSE (Mean Squared Error): `mean((pred - true)²)` over all genes and cells
- PCC (Pearson Correlation Coefficient): `pearsonr(true_flat, pred_flat)[0]` over all genes
- R² (Coefficient of Determination): `1 - (SS_res / SS_tot)` over all genes

**DE-Specific Metrics**: 
- MSE_DE: `mean((pred_DE - true_DE)²)` over top-20 DE genes per perturbation (selected by absolute log2 fold change)
- PCC_DE: `pearsonr(true_DE_flat, pred_DE_flat)[0]` over top-20 DE genes per perturbation, averaged across perturbations
- R²_DE: `1 - (SS_res_DE / SS_tot_DE)` over top-20 DE genes per perturbation, averaged across perturbations

**Additional Biological Validation Metrics**: 
- Pathway enrichment correlation: Correlation between predicted and observed pathway enrichment scores (additional validation, not core metric)
- Dose-response curve accuracy: Accuracy of predicted dose-response relationships (additional validation, not core metric)
- Cell-line-specific performance: Performance separately for each cell line (can be computed by subsetting results)

**Biological Metrics**: 
- Pathway enrichment scores (GSEA, GSVA)
- Gene set overlap (Jaccard, precision, recall)
- Mechanism-specific effect accuracy

**Validation Metrics**: 
- Held-out compound performance
- Held-out cell line performance
- Cross-validation performance
- Dose-response validation

#### Validation Strategy

**Cross-Validation**: 
- K-fold cross-validation within training compounds
- Stratified by compound, dose, and cell line
- Ensures robust performance estimation

**Biological Validation**: 
- Pathway enrichment analysis
- Dose-response curve validation
- Mechanism-specific effect validation
- Literature-based validation

**Independent Testing**: 
- Held-out compound evaluation (unseen drugs)
- Held-out cell line evaluation (unseen tissues)
- Ensures true generalization assessment

**Robustness Testing**: 
- Performance across different preprocessing choices
- Performance across different random seeds
- Performance across different train-test splits
- Ensures robustness to experimental variations

#### Benchmarks

**Baseline Models**: 
- CPA (Compositional Perturbation Autoencoder)
- cycleCDR
- CINEMA-OT
- ContrastiveVI
- scGen

**State-of-the-Art**: 
- Recent methods from single-cell perturbation prediction literature
- Methods from related fields (drug response prediction, perturbation prediction)

**Biological Benchmarks**: 
- Pathway enrichment accuracy
- Dose-response curve accuracy
- Mechanism-specific effect accuracy

**Performance Targets**: 
- MSE < 0.5 (normalized expression space)
- Global PCC > 0.7
- PCC_DE > 0.8 (computed over top-20 DE genes per perturbation)
- R² > 0.5 (indicates model explains >50% of variance)
- R²_DE > 0.6 (indicates model explains >60% of variance in key response genes)

### Performance Analysis

#### Model Performance

**Accuracy Metrics**: 
- MSE: Typically 0.3-0.8 (normalized expression space)
- PCC: Typically 0.6-0.8 (global, computed over all genes)
- PCC_DE: Typically 0.7-0.9 (computed over top-20 DE genes per perturbation)
- R²: Typically 0.4-0.7 (global, computed over all genes)
- R²_DE: Typically 0.5-0.8 (computed over top-20 DE genes per perturbation)

**Biological Relevance**: 
- Pathway enrichment correlation: Typically 0.5-0.7
- Dose-response accuracy: Varies by method and compound
- Mechanism-specific effects: Varies by method

**Generalization**: 
- Held-out compound performance: Typically 10-20% lower than training performance
- Held-out cell line performance: Typically 15-25% lower than training performance
- Cross-validation performance: Typically similar to training performance

**Robustness**: 
- Performance across different preprocessing choices: Typically stable (±5%)
- Performance across different random seeds: Typically stable (±3%)
- Performance across different train-test splits: Typically stable (±5%)

#### Comparative Analysis

**Baseline Comparison**: 
- CPA typically outperforms scGen on large-scale datasets
- cycleCDR typically outperforms CPA on generalization tasks
- CINEMA-OT typically outperforms others on interpretability tasks
- ContrastiveVI typically outperforms others on robustness tasks

**State-of-the-Art Comparison**: 
- Recent methods show 10-20% improvement over early baselines
- Improvements primarily in generalization and interpretability
- Computational efficiency improvements vary by method

**Biological Comparison**: 
- Methods with explicit biological constraints (e.g., pathway structure) show better biological interpretability
- Methods with dose-response modeling show better pharmacological plausibility
- Methods with attention mechanisms show better interpretability

**Computational Comparison**: 
- CPA: Moderate training time (1-3 days), moderate memory (8-16 GB)
- cycleCDR: Moderate training time (1-3 days), moderate memory (8-16 GB)
- CINEMA-OT: High training time (3-7 days), high memory (16-32 GB)
- ContrastiveVI: Moderate training time (1-3 days), moderate memory (8-16 GB)
- scGen: Low training time (<1 day), low memory (4-8 GB)

#### Strengths and Weaknesses

**Strengths**: 
- CPA: Compositionality, dose modeling, interpolation capability
- cycleCDR: Baseline preservation, generalization to novel drugs
- CINEMA-OT: Causal inference framework, individual treatment effects
- ContrastiveVI: Robustness to noise, separation of treatment effects
- scGen: Interpretability, simplicity, cross-cell-type generalization

**Weaknesses**: 
- CPA: Adversarial training instability, high-dimensional output challenges
- cycleCDR: Cycle loss collapse, loss weighting sensitivity
- CINEMA-OT: Computational cost, scalability limitations
- ContrastiveVI: Limited dose modeling, pretraining requirements
- scGen: Scalability limitations, lack of dose encoding

**Biological Insights**: 
- Baseline conditioning is essential for accurate predictions
- Dose-response modeling improves pharmacological plausibility
- Attention mechanisms improve interpretability
- Compositional designs align with biological reality

**Technical Limitations**: 
- High-dimensional outputs require efficient architectures
- Sparse data requires robust loss functions
- Batch effects require correction methods
- Computational scalability requires optimization

### Improvement Suggestions

#### Model Enhancements

**Architectural Improvements**: 
- Combine conditional transformers with graph-regularized VAEs for improved gene-gene interaction modeling
- Integrate contrastive pretraining for better baseline-perturbation disentanglement
- Use attention mechanisms to focus on relevant genes and pathways
- Implement modular architectures for flexible experimentation

**Training Optimizations**: 
- Use zero-inflated or negative binomial loss functions for sparse data
- Implement curriculum learning for gradual complexity increase
- Use data augmentation for improved generalization
- Optimize hyperparameters using automated search (e.g., Optuna)

**Biological Interpretability**: 
- Implement attention mechanisms for gene-level interpretability
- Use pathway enrichment analysis for pathway-level interpretability
- Visualize dose-response curves for key genes
- Provide mechanism-specific effect explanations

**Computational Efficiency**: 
- Use efficient data loading (e.g., HDF5, memory mapping)
- Implement gradient checkpointing for memory efficiency
- Use mixed-precision training for speedup
- Optimize architectures for GPU acceleration

#### Implementation Optimizations

**Code Optimizations**: 
- Vectorize operations for efficiency
- Use efficient data structures (e.g., sparse matrices)
- Optimize memory usage (e.g., gradient checkpointing)
- Parallelize data loading and preprocessing

**Pipeline Improvements**: 
- Standardize preprocessing pipelines
- Automate hyperparameter optimization
- Implement reproducible training procedures
- Create modular evaluation frameworks

**Validation Enhancements**: 
- Implement comprehensive evaluation pipelines
- Automate cross-validation procedures
- Create visualization tools for results
- Generate standardized evaluation reports

**Deployment Considerations**: 
- Create user-friendly APIs
- Implement model versioning
- Provide documentation and tutorials
- Ensure reproducibility across environments

#### Future Work

**Research Directions**: 
- Develop methods for multi-timepoint prediction
- Extend to multi-modal data (RNA + protein + chromatin)
- Improve generalization to unseen compounds and cell lines
- Enhance interpretability through attention and pathway analysis

**Biological Validation**: 
- Validate predictions on independent experimental data
- Compare pathway enrichment with known drug mechanisms
- Test dose-response relationships for known compounds
- Evaluate cell-line-specific effects

**Clinical Translation**: 
- Extend to patient-derived cell lines
- Incorporate patient-specific factors (genetics, disease state)
- Develop methods for personalized drug response prediction
- Validate on clinical trial data (if available)

**Technical Advancements**: 
- Develop more efficient architectures for large-scale datasets
- Improve handling of sparse and noisy data
- Enhance batch correction methods
- Develop methods for multi-task learning

### Detailed Baseline Method Analysis

### 1. Compositional Perturbation Autoencoder (CPA)

- **Approach**: VAE‐style autoencoder that disentangles baseline state and perturbation effect embeddings, allowing in silico generation of unmeasured combinations 
    
- **Strengths**:
    
    - Captures compositionality (drugs + doses + time).
        
    - Demonstrates strong interpolation to unseen dosages and species
        
- **Limitations**:
    
    - Relies on adversarial disentanglement which can be unstable.
        
    - Struggles with extremely high‐dimensional outputs without careful regularization.
        

### 2. cycleCDR

- **Approach**: Cycle‐consistency framework mapping control → perturbed and back, enforcing invertibility for robust counterfactual prediction 
    
- **Strengths**:
    
    - Encourages preservation of baseline states.
        
    - Improves generalization to novel drugs.
        
- **Limitations**:
    
    - Cycle losses can collapse when perturbations induce large shifts.
        
    - Requires careful weighting of cycle vs. reconstruction losses.
        

### 3. CINEMA-OT

- **Approach**: Causal‐inference and **Optimal Transport** to separate confounding variation from true perturbation effects, yielding counterfactual cell‐pairs for analysis 
    
- **Strengths**:
    
    - Theoretically grounded in causal inference.
        
    - Provides individual treatment‐effect estimates.
        
- **Limitations**:
    
    - Scalability issues for large perturbation panels.
        
    - Requires a good proxy for confounders to deconfound effectively.
        

### 4. ContrastiveVI

- **Approach**: Variational model that learns shared vs. salient latent spaces via **Contrastive Learning**, isolating perturbation‐specific signals
    
- **Strengths**:
    
    - Improves separation of treatment effects from baseline variation.
        
    - Demonstrates robustness to technical noise.
        
- **Limitations**:
    
    - Doesn’t explicitly model dose or continuous perturbation attributes.
        
    - Pretraining on control vs. treated pairs only.
        

### 5. scGen & Related Embedding Arithmetic

- **Approach**: Pretrain an autoencoder on control data, then perform latent‐space vector arithmetic (adding drug effect vector) to predict perturbed states
    
- **Strengths**:
    
    - Simplicity and interpretability of latent arithmetic.
        
    - Shown to generalize across cell types and unseen conditions.
        
- **Limitations**:
    
    - Best suited to small perturbation sets; scaling to hundreds of drugs is challenging.
        
    - Lacks explicit dose encoding; treats all perturbations as categorical shifts.
        

These baselines inform new-method designs that combine **conditional transformers**, **graph‐regularized VAEs**, and **contrastive pretraining** to jointly model baseline state, drug identity, and dose—while scaling to ∼650 000 cells and ∼188 compounds for robust generalization.

### Summary

The baseline model analysis reveals key insights for drug perturbation prediction:

- **Baseline conditioning is essential**: Models that condition on baseline expression consistently outperform those that do not, as cell-state-dependent drug responses require baseline context for accurate prediction.

- **Dose-response modeling improves accuracy**: Explicit dose encoding enables better dose-response prediction, as pharmacological dose-response relationships are fundamental to drug effects.

- **Attention mechanisms enhance interpretability**: Gene-level and pathway-level attention provide biological insights, enabling identification of key genes and pathways driving predictions.

- **Compositional designs align with biology**: Methods that model drug + dose + time compositionally better capture biological reality, as drug effects are inherently compositional.

- **Generalization requires robust architectures**: Methods with better generalization (e.g., cycleCDR, ContrastiveVI) use robust training strategies, including cycle consistency and contrastive learning.

- **Computational efficiency is critical**: Large-scale datasets (∼650,000 cells, 188 compounds) require efficient architectures and training procedures, with computational costs varying significantly across methods.

- **Biological validation is essential**: All methods should be validated on held-out compounds and cell lines, with pathway enrichment and dose-response validation ensuring biological plausibility.

These insights guide the design of new methods that combine the strengths of existing approaches while addressing their limitations, ultimately enabling accurate, interpretable, and generalizable drug perturbation prediction.