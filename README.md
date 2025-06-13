# AI Multimedia Lab at ImageCLEFmedical GANs 2025: Identifying Real-Image Usage in Generated Medical Images
Notebook for the ImageCLEF Lab at CLEF 2025
Alexandra-Georgiana Andrei1, Mihai Gabriel Constantin, Mihai Dogariu, Liviu-Daniel Stefan1 and Bogdan Ionescu
AI Multimedia Lab, National University of Science and Technology Politehnica Bucharest, Romania

Abstract
This paper presents the participation of AI Multimedia Lab in the third edition of the 2025 ImageCLEFmedical GANs task, which investigates privacy and security concerns around generated synthetic medical images. This edition, the challenge comprises two complementary subtasks: (1) Detect which real images were used to train a GAN based on given synthetic outputs and (2) Attribute each synthetic image to its specific real-image subset  of origin. We present our team’s approach, which combines traditional deep learning techniques on two-step pipeline - feature extraction followed by clustering - for subtask 2, and Siamese Neural Networks for both subtasks. Evaluated on benchmark testing datasets of real and synthetic lung CT slices, our Siamese-based method achieved
a Cohen’s kappa of 0.036 for Subtask 1 and 99.04% accuracy for Subtask 2. Finally, we discuss the strengths and limitations of our methods and outline directions for improving the detection of training-data “fingerprints” in GAN-generated medical images.

Keywords: synthetic medical data, Generative Adversarial Networks, data augmentation, ImageCLEFmedical GANs, Image-
CLEFbenchmarking lab,
