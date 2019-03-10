# EEG_classification
Description of the approach : https://towardsdatascience.com/sleep-stage-classification-from-single-channel-eeg-using-convolutional-neural-networks-5c710d92d38e


Sleep Stage Classification from Single Channel EEG using Convolutional Neural
Networks

*****

<span class="figcaption_hack">Photo by [Paul
M](https://unsplash.com/photos/7i9yLoUgoP8?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText)
on
[Unsplash](https://unsplash.com/search/photos/owl?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText)</span>

Quality Sleep is an important part of a healthy lifestyle as lack of it can
cause a list of
[issues](https://www.webmd.com/sleep-disorders/features/10-results-sleep-loss#1)
like a higher risk of cancer and chronic fatigue. This means that having the
tools to automatically and easily monitor sleep can be powerful to help people
sleep better.<br> Doctors use a recording of a signal called EEG which measures
the electrical activity of the brain using an electrode to understand sleep
stages of a patient and make a diagnosis about the quality if their sleep.

In this post we will train a neural network to do the sleep stage classification
automatically from EEGs.

### **Data**

In our input we have a sequence of 30s epochs of EEG where each epoch has a
label [{“W”, “N1”, “N2”, “N3”,
“REM”}](https://en.wikipedia.org/wiki/Sleep_cycle).

<span class="figcaption_hack">Fig 1 : EEG Epoch</span>

<span class="figcaption_hack">Fig 2 : Sleep stages through the night</span>

This post is based on a publicly available EEG Sleep data (
[Sleep-EDF](https://www.physionet.org/physiobank/database/sleep-edfx/) ) that
was done on 20 subject, 19 of which have 2 full nights of sleep. We use the
pre-processing scripts available in this
[repo](https://github.com/akaraspt/deepsleepnet) and split the train/test so
that no study subject is in both at the same time.

The general objective is to go from a 1D sequence like in fig 1 and predict the
output hypnogram like in fig 2.

### Model Description

Recent approaches [[1]](https://arxiv.org/pdf/1703.04046.pdf) use a sub-model
that encodes each epoch into a 1D vector of fixed size and then a second
sequential sub-model that maps each epoch’s vector into a class from [{“W”,
“N1”, “N2”, “N3”, “REM”}](https://en.wikipedia.org/wiki/Sleep_cycle).

Here we use a 1D CNN to encode each Epoch and then another 1D CNN or LSTM that
labels the sequence of epochs to create the final
[hypnogram](https://en.wikipedia.org/wiki/Hypnogram). This allows the prediction
for an epoch to take into account the context.

<span class="figcaption_hack">Sub-model 1 : Epoch encoder</span>

<span class="figcaption_hack">Sub-model 2 : Sequential model for epoch classification</span>

The full model takes as input the sequence of EEG epochs ( 30 seconds each)
where the sub-model 1 is applied to each epoch using the TimeDistributed Layer
of [Keras](https://keras.io/) which produces a sequence of vectors. The sequence
of vectors is then fed into a another sub-model like an LSTM or a CNN that
produces the sequence of output labels.<br> We also use a linear Chain
[CRF](https://en.wikipedia.org/wiki/Conditional_random_field) for one of the
models and show that it can improve the performance.

### Training Procedure

The full model is trained end-to-end from scratch using Adam optimizer with an
initial learning rate of 1e⁻³ that is reduced each time the validation accuracy
plateaus using the ReduceLROnPlateau Keras Callbacks.

<span class="figcaption_hack">Accuracy Training curves</span>

### Results

We compare 3 different models :

* CNN-CNN : This ones used a 1D CNN for the epoch encoding and then another 1D CNN
for the sequence labeling.
* CNN-CNN-CRF : This model used a 1D CNN for the epoch encoding and then a 1D
CNN-CRF for the sequence labeling.
* CNN-LSTM : This ones used a 1D CNN for the epoch encoding and then an LSTM for
the sequence labeling.

We evaluate each model on an independent test set and get the following results
:

* CNN-CNN : F1 = 0.81, ACCURACY = 0.87
* CNN-CNN-CRF : F1 = 0.82, ACCURACY =0.89
* CNN-LSTM : F1 = 0.71, ACCURACY = 0.76

The CNN-CNN-CRF outperforms the two other models because the CRF helps learn the
transition probabilities between classes. The LSTM based model does not work as
well because it is most sensitive to hyper-parameters like the optimizer and the
batch size and requires extensive tuning to perform well.

<span class="figcaption_hack">Ground Truth Hypnogram</span>

<span class="figcaption_hack">Predicted Hypnogram using CNN-CNN-CRF</span>

Source code available here :
[https://github.com/CVxTz/EEG_classification](https://github.com/CVxTz/EEG_classification)

I look forward to your suggestions and feedback.

[[1] DeepSleepNet: a Model for Automatic Sleep Stage Scoring based on Raw
Single-Channel EEG](https://arxiv.org/pdf/1703.04046.pdf)

