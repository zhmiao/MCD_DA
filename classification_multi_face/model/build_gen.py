import model.svhn2mnist as svhn2mnist
import model.usps as usps
import model.syn2gtrsb as syn2gtrsb
# import model.syndig2svhn as syndig2svhn

def Generator(source, target, pixelda=False):
    if source == 'usps' or target == 'usps':
        return usps.Feature()
    elif source == 'svhn':
        return svhn2mnist.Feature()
    elif source == 'synth':
        return syn2gtrsb.Feature()
    elif source == 'svhn_bal':
        return svhn2mnist.Feature()


def Classifier(source, target):
    if source == 'usps' or target == 'usps':
        return usps.Predictor()
    if source == 'svhn':
        return svhn2mnist.Predictor()
    if source == 'synth':
        return syn2gtrsb.Predictor()
    if source == 'svhn_bal':
        return svhn2mnist.Predictor()

