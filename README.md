<div align="center">

# Vision Supervised Radiology Report Generation 🩺☤🧑🏻‍⚕️
*X-Ray Insights, AI-Generated, Expert Approved*

</div>

### Project Overview

Accurate and timely radiology reporting is critical for elective clinical diagnosis, yet manual report generation is time-consuming and prone to variability.<br/><br/>
This project proposes a robust **Multimodal Sequence-toSequence (Seq2Seq)** framework designed for complex crossmodal tasks such as medical image captioning or visual-linguistic reasoning. The architecture leverages a dual-stream feature extraction process: utilizing **BioClinicalBERT** for deep textual encoding and **Vision Transformers (ViT) or ResNet-50** for highdimensional visual representation. These unimodal features are aligned into a shared **[1 * 768 ]** latent space and integrated through a **cross-modal interaction layer**, allowing the model to capture **bidirectional dependencies (Text to Vision and Vision to Text)**.<br/><br/>
The fused multimodal embeddings are then fed into a **Transformerbased Encoder-Decoder system.** The Encoder generates a unified contextual representation, while the Decoder employs an autoregressive **Next Word Prediction (NWP)** strategy—facilitated by Teacher Forcing and **self-attention mechanisms** to generate accurate, context-aware natural language sequenProblem Statementces. This integrated approach ensures the model electively bridges the semantic gap between visual data and clinical textual sequences, providing a scalable solution for multimodal AI applications.<br/><br/>

---

### Technical Domain Specification
 &nbsp;&nbsp;&nbsp;&nbsp; ***A.Hardware & OS*** :<br>
     &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; - **Operating System**: Windows 10/11 or Ubuntu (Linux) 20.04+.<br>
     &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; - **GPU (Crucial)**: NVIDIA RTX 30-series or higher (8GB+ VRAM) to handle Transformer training.<br>
     &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; - **RAM**: Minimum 16GB<br>
&nbsp;&nbsp;&nbsp;&nbsp; ***B. Software & Programming*** :<br>
     &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; - **Language**: Python 3.8+ (The standard for AI). <br>
     &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; - **Development Env**: VS Code or PyCharm; Jupyter Notebooks/Google Colab for experimentation.  <br>
&nbsp;&nbsp;&nbsp;&nbsp; ***C. Libraries & Frameworks***<br/>
     &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; - **Deep Learning**: PyTorch or TensorFlow (Keras). <br>
     &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; - **Transformers**: Hugging Face transformers library (for BERT, ViT, and T5/BART).<br>
     &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; - **Computer Vision**: torchvision, OpenCV, or PIL. <br>
     &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; - **Data Handling**: NumPy and Pandas.<br>
     &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; - **Deployment**: Flask or FastAPI (if creating a web interface).<br>

### Problem Statement

&nbsp;&nbsp;&nbsp;&nbsp; **A. Multi-modal Medical Information :** <br/>
Radiology data is multimodal, meaning it contains :<br/> 1. Images (Vision) → Chest X-ray images <br/>2.Text (Language) → Radiology reports written by
doctors ,<br/>

**1. Image Modality (I)** : Chest X-rays can have multiple views: PA (Postero-Anterior) , AP (Antero-Posterior) , LAT (Lateral) <br/> **So , I = {PA, AP, LAT}** <br/>
**2. Text Modality (RT – Radiology Text)** : A radiology report is not a single paragraph, it is structured into sections , MeSH , Problems , Image , Indication , Comparison , <br/> **So , RT = {SE₁, SE₂, ..., SEₖ}**

<br/> **B. Sentence-Level and Medical Entity Breakdown :** <br/>
Findings contain multiple sentences SE_findings = {S₁, S₂, ..., Sₚ} having dierent entities Diseases , Anatomical regions , Observations as Sᵢ = {ME₁, ME₂, ..., ME_q}

<br/>**C . Dataset Representation** : dataset D with two
components <br/>D = {M₁, M₂} where M₁ = Images (I) , M₂ = Reports (RT) Each training sample has Sᵢ → (RT, I)

<br/>**D. Model Pipeline**<br/>
**X-ray image (Vision) → Processed Image →Our Model →Feature Extraction → Modelling → Decision Making (Vision Encoding)**<br/>

**Report Text → Our Model →Feature Extraction RT (F.E) + I(F.E) → Multi-modal Integration / Cross-modal**<br/>

#### Interaction<br/>
The multi-modal radiology report summarization (MRRS) is a conditional generation problem over two input modalities(1), where a parameterized neural network tries to map an image- text pair to generate a concise summary that can maximize the likelihood of the ground truth impression section, and can be defined as follows:<br/> <img width="239" height="57" alt="image" src="https://github.com/user-attachments/assets/d5beee6e-f64d-42c8-8e54-22b7d10c879d" /><br/>

where, we seek the model parameters θ* that maximize the loglikelihood of the ground-truth target sequences S*, conditioned on the inputs, image I and text T, across all samples in our dataset D. Equivalently, this is the standard maximum likelihood estimation (MLE) objective adapted to conditional sequence modeling , θ* maximizes the sum of log-probabilities log P θ(S* / I, T).<br/>

### Architecture <br/>
<img width="471" height="569" alt="image" src="https://github.com/user-attachments/assets/fd5c6da5-e49b-40cb-91d7-f89ee4a60ea4" /><br/>


### Contribution & Related Works<br/>
Existing research has investigated vision–language models for generating chest X-ray radiology reports, focusing on improving image–text alignment and clinical accuracy.<br/>
<img width="850" height="379" alt="image" src="https://github.com/user-attachments/assets/fc0319f0-ef68-40c5-96f6-ef6f1460edd4" /><br/>
<img width="855" height="420" alt="image" src="https://github.com/user-attachments/assets/02b46ace-b79b-405e-ba02-dcfdb3785f73" /><br/>


### Gap Analysis <br/>
<img width="1243" height="699" alt="image" src="https://github.com/user-attachments/assets/9ecb86c3-8401-425f-b967-5e7f9218926a" /><br/>
<img width="1234" height="704" alt="image" src="https://github.com/user-attachments/assets/0bc59ed2-a755-475c-a516-90b92b459a02" /><br/>

### Dataset Description<br/>
<img width="1238" height="644" alt="image" src="https://github.com/user-attachments/assets/f5869cc9-1b18-4b49-9641-78efa7d08c0a" /><br/>

#### Image Dataset Analysis<br/>
<img width="692" height="783" alt="image" src="https://github.com/user-attachments/assets/d544e879-afd0-42b0-a722-0941b7d4ee59" /><br/>
<img width="692" height="801" alt="image" src="https://github.com/user-attachments/assets/30c39c94-d1aa-429e-9ea7-c938430ff8b6" /><br/>
##### Dataset Link : https://shorturl.at/WyjsQ <br/>

#### Projection Metadata Analysis<br/>
<img width="682" height="653" alt="image" src="https://github.com/user-attachments/assets/49197334-11d4-4f12-8d33-7ccec51c6ed8" /><br/>
##### Dataset Link : (indiana_projections.csv) https://shorturl.at/JsVZ2 <br/>

#### Reports Dataset<br/>
<img width="590" height="696" alt="image" src="https://github.com/user-attachments/assets/78775f7b-9776-48c2-b466-d38c42707927" /><br/>
##### Dataset Link : (indiana_reports.csv) : https://shorturl.at/G41rb<br/>

#### Image : Report Relationship <br/>
<img width="689" height="750" alt="image" src="https://github.com/user-attachments/assets/e9973b43-169a-419e-8202-c77942dd2618" /><br/>
<img width="342" height="657" alt="image" src="https://github.com/user-attachments/assets/5a38e966-874e-4850-9d36-9f4c3982656c" /><img width="508" height="354" alt="image" src="https://github.com/user-attachments/assets/6015f8eb-be79-421d-b5de-bc4a90ba8fe8" /><br/>

##### Image Report Analysis<br/>
<img width="356" height="664" alt="image" src="https://github.com/user-attachments/assets/b06a1b6d-7c67-4a64-ae22-0a9ad8694689" /><br/>
<img width="470" height="564" alt="image" src="https://github.com/user-attachments/assets/269d2246-5004-4f19-9e5e-cd3a16e8dd27" /><br/>

##### Visual & Clinical Diversity<br/>
<img width="451" height="592" alt="image" src="https://github.com/user-attachments/assets/ee7a344d-06aa-4b4d-bf56-0c8841a251dc" /><br/>












