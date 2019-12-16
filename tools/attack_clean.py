## attack.py -- generate audio adversarial examples
##
## Copyright (C) 2017, Nicholas Carlini <nicholas@carlini.com>.
##
## This program is licenced under the BSD 2-Clause licence,
## contained in the LICENCE file in this directory.

## TODO pad audio with one windowsize extra for stft

import numpy as np
import tensorflow as tf
import argparse
import librosa
from shutil import copyfile
from reweber import convolution as reweber
from scipy.ndimage import gaussian_filter

import scipy.io.wavfile as wav

import struct
import time
import os
import sys
from collections import namedtuple
sys.path.append("DeepSpeech")

try:
    import pydub
except:
    print("pydub was not loaded, MP3 compression will not work")

# Okay, so this is ugly. We don't want DeepSpeech to crash.
# So we're just going to monkeypatch TF and make some things a no-op.
# Sue me.
tf.load_op_library = lambda x: x
tmp = os.path.exists
os.path.exists = lambda x: True
class Wrapper:
    def __init__(self, d):
        self.d = d
    def __getattr__(self, x):
        return self.d[x]
class HereBeDragons:
    d = {}
    FLAGS = Wrapper(d)
    def __getattr__(self, x):
        return self.do_define
    def do_define(self, k, v, *x):
        self.d[k] = v
tf.app.flags = HereBeDragons()
import DeepSpeech
os.path.exists = tmp

# More monkey-patching, to stop the training coordinator setup
DeepSpeech.TrainingCoordinator.__init__ = lambda x: None
DeepSpeech.TrainingCoordinator.start = lambda x: None


from util.text import ctc_label_dense_to_sparse
from tf_logits import get_logits

# These are the tokens that we're allowed to use.
# The - token is special and corresponds to the epsilon
# value in CTC decoding, and can not occur in the phrase.
toks = " abcdefghijklmnopqrstuvwxyz'-"

def convert_mp3(new, length, filename = None):
    import pydub
    wav.write("/tmp/load.wav", 16000,
              np.array(np.clip(np.round(new[:length]),
                               -2**15, 2**15-1),dtype=np.int16))
    if(filename == None):
        pydub.AudioSegment.from_wav("/tmp/load.wav").export("/tmp/saved.mp3", format='mp3', bitrate='192k')
        raw = pydub.AudioSegment.from_mp3("/tmp/saved.mp3")
    else:
        pydub.AudioSegment.from_wav("/tmp/load.wav").export("/tmp/" + filename, format='mp3', bitrate='192k')
        raw = pydub.AudioSegment.from_mp3("/tmp/" + filename)
    mp3ed = np.array([struct.unpack("<h", raw.raw_data[i:i+2])[0] for i in range(0,len(raw.raw_data),2)], dtype=np.int16)[np.newaxis,:length]
    return mp3ed

def gauss2d(size=5, sigma=1):
    assert(size%2==1)
    gaussKernel = np.zeros(size)
    gaussKernel[size//2] = 1
    return(gaussian_filter(np.outer(gaussKernel, gaussKernel), sigma))

class Attack:
    def __init__(self, sess, loss_fn, phrase_length, max_audio_len, psdMaxes,
                     learning_rate=10, num_iterations=5000, window_size=2048,
                     step_per_window=4, batch_size=1, mp3=False, 
                     onlyCTC=True, audio=None, psdShape=None):
        """
        Set up the attack procedure.

        Here we create the TF graph that we're going to use to
        actually generate the adversarial examples.
        """
        
        self.sess = sess
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.batch_size = batch_size
        self.phrase_length = phrase_length
        self.max_audio_len = max_audio_len
        self.mp3 = mp3
        self.psdMaxes = psdMaxes
        self.window_size = window_size
        self.step_per_window = step_per_window
        
        # Create all the variables necessary
        # they are prefixed with qq_ just so that we know which
        # ones are ours so when we restore the session we don't
        # clobber them.
        
        frame_length = int(window_size)
        frame_step = int(window_size//step_per_window)
        fft_length = int(2**np.ceil(np.log2(frame_length)))
        sample_rate = 16000
        freq_res = sample_rate/window_size
        time_res = frame_step/(sample_rate/1000)
        sigma_time = 96. / time_res
        sigma_freq = 15.625 / freq_res
        
        self.regularizer = regularizer = tf.Variable(np.zeros((batch_size), dtype=np.float32), name='qq_regularizer')
        self.psyTh = psyTh = tf.Variable(np.zeros((batch_size, psdShape[0], psdShape[1]), dtype=np.float32), name='qq_psyTh')
        
        self.delta = delta = tf.Variable(np.zeros((batch_size, max_audio_len)).astype(np.float32)/2, name='qq_delta') name='qq_delta')
        self.mask = mask = tf.Variable(np.zeros((batch_size, max_audio_len), dtype=np.float32), name='qq_mask')
        self.original = original = tf.Variable(np.zeros((batch_size, max_audio_len), dtype=np.float32), name='qq_original')
        self.lengths = lengths = tf.Variable(np.zeros(batch_size, dtype=np.int32), name='qq_lengths')
        self.target_phrase = tf.Variable(np.zeros((batch_size, phrase_length), dtype=np.int32), name='qq_phrase')
        self.target_phrase_lengths = tf.Variable(np.zeros((batch_size), dtype=np.int32), name='qq_phrase_lengths')
        self.rescale = tf.Variable(np.zeros((batch_size,1), dtype=np.float32), name='qq_rescale')

        # Initially we bound the l_infty norm by 2000, increase this
        # constant if it's not big enough of a distortion for your dataset.
         
        if(loss_fn == 'CTC'):
            self.apply_delta = tf.clip_by_value(delta, -2000, 2000)*self.rescale
        elif(loss_fn == 'CTCPSYCLIP'):
            
            self.apply_delta = apply_delta = self.clipBatch(delta, psyTh, regularizer, psdMaxes, max_audio_len, window_size, step_per_window)
            
            self.new_input = new_input = self.apply_delta*mask + original
            
            #self.new_input = new_input = delta*mask + original
            
        # We set the new input to the model to be the above delta
        # plus a mask, which allows us to enforce that certain
        # values remain constant 0 for length padding sequences.
        
        if(loss_fn == 'CTC'):
            self.new_input = new_input = self.apply_delta*mask + original
        if(loss_fn == 'CTCPSYGRAD'):
            self.new_input = new_input = self.delta*mask + original

        # We add a tiny bit of noise to help make sure that we can
        # clip our values to 16-bit integers and not break things.
        if(loss_fn == 'CTC'):
            noise = tf.random_normal(new_input.shape,
                                     stddev=2)
            pass_in = tf.clip_by_value(new_input+noise, -2**15, 2**15-1)
 
        # Feed this final value to get the logits.
        self.logits = logits = get_logits(new_input, lengths)

        # And finally restore the graph to make the classifier
        # actually do something interesting.
        saver = tf.train.Saver([x for x in tf.global_variables() if 'qq' not in x.name])
        saver.restore(sess, "models/session_dump")

        self.loss_fn = loss_fn
        
        
        if loss_fn == "CTC":
            target = ctc_label_dense_to_sparse(self.target_phrase, self.target_phrase_lengths, batch_size)
            
            ctcLoss = tf.nn.ctc_loss(labels=tf.cast(target, tf.int32),
                                     inputs=logits, sequence_length=lengths)

            # Slight hack: an infinite l2 penalty means that we don't penalize l2 distortion
            # The code runs faster at a slight cost of distortion, and also leaves one less
            # paramaeter that requires tuning.
            if not onlyCTC:
                loss = tf.reduce_mean((self.new_input-self.original)**2,axis=1)/regularizer + ctcLoss
            else:
                loss = ctcLoss
            self.expanded_loss = tf.constant(0)
        elif loss_fn == "CTCPSYCLIP":
            target = ctc_label_dense_to_sparse(self.target_phrase, self.target_phrase_lengths, batch_size)
            
            ctcLoss = tf.nn.ctc_loss(labels=tf.cast(target, tf.int32),
                                     inputs=logits, sequence_length=lengths)
            loss = ctcLoss
            self.expanded_loss = tf.constant(0)
        elif loss_fn == "CW":
            raise NotImplemented("The current version of this project does not include the CW loss function implementation.")
        else:
            raise
            
        self.deltaPSD = deltaPSD = tfPSD(self.new_input-self.original, window_size, step_per_window, self.psdMaxes)
        psyLoss = tf.reduce_max(deltaPSD - self.psyTh, axis=[1,2])
        self.loss = loss
        self.psyLoss = tf.transpose(psyLoss)
        self.ctcLoss = ctcLoss
        
        # Set up the Adam optimizer to perform gradient descent for us
        start_vars = set(x.name for x in tf.global_variables())
        optimizer = tf.train.AdamOptimizer(learning_rate)

        grad,var = optimizer.compute_gradients(self.loss, [delta])[0]
        self.train = optimizer.apply_gradients([(grad,var)])
        
        end_vars = tf.global_variables()
        new_vars = [x for x in end_vars if x.name not in start_vars]
        
        sess.run(tf.variables_initializer(new_vars+[delta]))

        # Decoder from the logits, to see how we're doing
        self.decoded, _ = tf.nn.ctc_beam_search_decoder(logits, lengths, merge_repeated=False, beam_width=100)

    def clipBatch(self, delta, psyTh, regularizer, psdMaxes, max_audio_len, window_size, step_per_window):
        deltaShape = delta.shape
        psdShape = psyTh.shape
        batch_size = deltaShape[0]
        frame_length = int(window_size)
        frame_step = int(window_size//step_per_window)
        fft_length = int(2**np.ceil(np.log2(frame_length)))
    
        paddingRounding = max_audio_len%frame_step
        paddingAudio = tf.constant(np.array([[0,0],[frame_length,frame_length]]), dtype=tf.int32)
        paddingStft = tf.constant(np.array([[0,0], [step_per_window, step_per_window],[0,0]]), dtype=tf.int32)
        inverse_window_fn = tf.contrib.signal.inverse_stft_window_fn(frame_step)
        gauss2dFilter = tf.constant(gauss2d(101,[2,6]).reshape(101,101,1,1), dtype=tf.float32)
        deltaPad = tf.reshape(tf.pad(delta, paddingAudio, "CONSTANT", constant_values=tf.constant(0., dtype=tf.float32)),[batch_size, -1])
        psyThPad = tf.pad(psyTh, paddingStft, "CONSTANT", constant_values=tf.constant(0., dtype=tf.float32))
        deltaPSD = tfPSD(deltaPad, window_size, step_per_window, psdMaxes)
       
        diffMax = tf.reduce_max(deltaPSD - (psyThPad + tf.reshape(regularizer, [batch_size, 1, 1])))
        
        cond = lambda argDiffMax,argDeltaPad: tf.reduce_max(argDiffMax-(tf.reshape(self.regularizer, [-1,1,1]))) > 0
        cond = lambda argDiffMax, argDeltaPad, count: count < 5
        def body(argDiffMax,argDeltaPad,count):
            deltaPSD = tfPSD(argDeltaPad, window_size, step_per_window, psdMaxes)
            diff1 = tf.maximum(tf.nn.softplus(deltaPSD - (psyThPad + tf.reshape(self.regularizer-0.5, [-1,1,1]))),0.1)
            argDiffMax = tf.reduce_max(diff1)
            diff4 = tf.reshape(
                tf.nn.conv2d(tf.reshape(
                        diff1, 
                        [batch_size, psdShape[1]+2*step_per_window,psdShape[2],1]
                    ),
                    gauss2dFilter,
                    [1,1,1,1],
                    'SAME'),
                [batch_size, psdShape[1]+2*step_per_window,psdShape[2]])
            diff5 = diff4/tf.reduce_max(diff4)*argDiffMax
            deltaStft = tf.contrib.signal.stft(
                argDeltaPad,
                frame_length,
                frame_step,
                fft_length)
            argDeltaPad = tf.reshape(
                tf.pad(
                    tf.contrib.signal.inverse_stft(
                        deltaStft/tf.cast(tf.pow(10.0,diff5/20.0), tf.complex64),
                        frame_length, frame_step, fft_length, window_fn=inverse_window_fn
                    ),
                    tf.constant([[0,0],[0,paddingRounding]])
                ), 
                deltaPad.shape)
            argDiffMax = tf.reduce_max(deltaPSD - (psyThPad + tf.reshape(self.regularizer, [-1,1,1])))
            
            count += 1
            return (argDiffMax, argDeltaPad, count)
        
        (diffMax,deltaPad, count) = tf.while_loop(cond,body,(diffMax,deltaPad, 0))
        reshaped_inv_stft = tf.reshape(deltaPad,[batch_size, (max_audio_len+2*frame_length)])
        deltaRet = tf.reshape(tf.slice(reshaped_inv_stft,[0, frame_length],[batch_size, max_audio_len]), delta.shape)
        return deltaRet

    def clip(self, delta, psyTh, regularizer, psdMax, max_audio_len, window_size, step_per_window):
        deltaShape = delta.shape
        psdShape = psyTh.shape
        delta = tf.expand_dims(delta, 0)
        psyTh = tf.expand_dims(psyTh, 0)
        batch_size = 1
        frame_length = int(window_size)
        frame_step = int(window_size//step_per_window)
        fft_length = int(2**np.ceil(np.log2(frame_length)))
        sample_rate = 16000
        freq_res = sample_rate/window_size
        time_res = frame_step/(sample_rate/1000)
    
        paddingRounding = max_audio_len%frame_step
        paddingAudio = tf.constant(np.array([[0,0],[frame_length,frame_length]]), dtype=tf.int32)
        paddingStft = tf.constant(np.array([[0,0], [step_per_window, step_per_window],[0,0]]), dtype=tf.int32)
        inverse_window_fn = tf.contrib.signal.inverse_stft_window_fn(frame_step)
        gauss2dFilter = tf.constant(gauss2d(101,[2,6]).reshape(101,101,1,1), dtype=tf.float32)
        
        deltaPad = tf.reshape(tf.pad(delta, paddingAudio, "CONSTANT", constant_values=tf.constant(0., dtype=tf.float32)),[1, -1])
        
        psyThPad = tf.pad(psyTh, paddingStft, "CONSTANT", constant_values=tf.constant(0., dtype=tf.float32))
        deltaPSD = tfPSD(deltaPad, window_size, step_per_window, psdMax)
       
        diffMax = tf.reduce_max(deltaPSD - (psyThPad + tf.reshape(regularizer, [1, 1, 1])))
        
        cond = lambda argDiffMax,argDeltaPad, argDeltaPSD: argDiffMax>regularizer-0.5
        def body(argDiffMax,argDeltaPad, argDeltaPSD):
            argDeltaPSD = tf.expand_dims(argDeltaPSD, 0)
            diff = tf.maximum(tf.nn.relu(argDeltaPSD - (psyThPad + regularizer-0.5)),0.1)
            argDiffMax = tf.reduce_max(diff) 
            diff = tf.reshape(
                tf.nn.conv2d(tf.reshape(diff, [1,psdShape[0]+2*step_per_window,psdShape[1],1]),gauss2dFilter,[1,1,1,1], 'SAME'), 
                [1, psdShape[0]+2*step_per_window,psdShape[1]])
            diff = diff/tf.reduce_max(diff)*argDiffMax
            deltaStft = tf.contrib.signal.stft(argDeltaPad, frame_length, frame_step, fft_length)
            argDeltaPad = tf.reshape(tf.pad(
                tf.contrib.signal.inverse_stft(deltaStft/tf.cast(tf.pow(10.0,diff/20.0), tf.complex64), frame_length, frame_step, fft_length, window_fn=inverse_window_fn),
                tf.constant([[0,0],[0,paddingRounding]])), deltaPad.shape)
            argDeltaPSD = tfPSD(argDeltaPad, window_size, step_per_window, psdMax)
            argDiffMax = tf.reduce_max(argDeltaPSD - (psyThPad + regularizer))
            
            argDeltaPad = tf.squeeze(argDeltaPad)
            argDeltaPSD = tf.squeeze(argDeltaPSD)
            return (argDiffMax, argDeltaPad, argDeltaPSD)
        
        (diffMax,deltaPad,deltaPSD) = tf.while_loop(cond,body,(diffMax,tf.squeeze(deltaPad),tf.squeeze(deltaPSD)))
        
        reshaped_inv_stft = tf.reshape(deltaPad,[batch_size, (max_audio_len+2*frame_length)])
        delta = tf.reshape(tf.slice(reshaped_inv_stft,[0, frame_length],[batch_size, max_audio_len]), delta.shape)
        return tf.squeeze(delta)
        

    def attack(self, audio, psyTh, lengths, target, regularizer = [0], finetune=None):
        sess = self.sess

        # Initialize all of the variables
        # TODO: each of these assign ops creates a new TF graph
        # object, and they should be all created only once in the
        # constructor. It works fine as long as you don't call
        # attack() a bunch of times.
        sess.run(tf.variables_initializer([self.delta]))
        sess.run(self.regularizer.assign(np.array(regularizer).reshape((-1))))
        #sess.run(self.power.assign(np.ones(1)))
        sess.run(self.psyTh.assign(np.array(psyTh)))
        
        sess.run(self.original.assign(np.array(audio)))
        sess.run(self.lengths.assign((np.array(lengths)-1)//320))
        sess.run(self.mask.assign(np.array([[1 if i < l else 0 for i in range(self.max_audio_len)] for l in lengths])))
        sess.run(self.target_phrase_lengths.assign(np.array([len(x) for x in target])))
        sess.run(self.target_phrase.assign(np.array([list(t)+[0]*(self.phrase_length-len(t)) for t in target])))
        c = np.ones((self.batch_size, self.phrase_length))
        sess.run(self.rescale.assign(np.ones((self.batch_size,1))))

        # Here we'll keep track of the best solution we've found so far
        final_deltas = [None]*self.batch_size

        if finetune is not None and len(finetune) > 0:
            sess.run(self.delta.assign(finetune-audio))
        
        # We'll make a bunch of iterations of gradient descent here
        start = time.time()
        MAX = self.num_iterations
        
        bestCTC = [float('inf')]*self.batch_size
        bestPSY = [float('inf')]*self.batch_size
        b = False
        count = [0]*self.batch_size
        for i in range(MAX):
            iteration = i
            now = time.time()
            # Print out some debug information every 10 iterations.
            if i%10 == 0:
                (d, d2,
                    plWAV, loss, r_logits, 
                    new_input, r_out, regularizer, rescale) = sess.run((
                        self.delta, self.apply_delta, 
                        self.psyLoss, self.loss, self.logits, 
                        self.new_input, self.decoded, self.regularizer, self.rescale))
                lst = [(r_out, r_logits, plWAV, loss, regularizer)]
                if self.mp3:
                    mp3ed = []
                    for ii in range(len(new_input)):
                        mp3ed.append(convert_mp3(new_input[ii], max(lengths)))
                    mp3ed = np.concatenate(mp3ed, axis = 0)
                    print(mp3ed.shape)
                    mp3_out, mp3_logits, plMP3, loss = sess.run((
                        self.decoded, self.logits, self.psyLoss, self.loss),
                        {self.new_input: mp3ed})
                    lst = [(mp3_out, mp3_logits, plMP3, loss, regularizer)]

                for out, logits, pl, loss, regularizer in lst:
                    chars = out[0].values

                    res = np.zeros(out[0].dense_shape)+len(toks)-1
                
                    for ii in range(len(out[0].values)):
                        x,y = out[0].indices[ii]
                        res[x,y] = out[0].values[ii]

                    # Here we print the strings that are recognized.
                    res = ["".join(toks[int(x)] for x in y).replace("-","") for y in res]
                    #if("".join([toks[x] for x in target[0]]) != "".join(res)):
                        
                    #print('ctc loss   ', cl)
                    print('psyLoss    ', pl)
                    print('bestPsy    ', bestPSY)
                    print('loss       ', loss)
                    if(self.loss_fn=='CTCPSYCLIP'):
                        print('regularizer', regularizer.reshape(-1))
                    else:
                        print('regularizer', 2000*rescale.reshape(-1))
                    #print("\n".join(res))
                    
                    # And here we print the argmax of the alignment.
                    res2 = np.argmax(logits,axis=2).T
                    res2 = ["".join(toks[int(x)] for x in y[:(l-1)//320]) for y,l in zip(res2,lengths)]
                    #print("\n".join(res2))


            if self.mp3:
                new = sess.run(self.new_input)
                mp3ed = []
                for ii in range(len(new_input)):
                    mp3ed.append(convert_mp3(new[ii], max(lengths))[0])
                mp3ed = np.array(mp3ed)
                feed_dict = {self.new_input: mp3ed}
            else:
                feed_dict = {}
                
            # Actually do the optimization ste
            (train) = sess.run((self.train),feed_dict)
            
            # Report progress
            print('i: ', i, time.time()-start)

            logits = np.argmax(r_logits, axis=2).T
            if(i%10==0):
                # Every 100 iterations, check if we've succeeded
                # if we have (or if it's the final epoch) then we
                # should record our progress and decrease the
                # rescale constant.
                for ii in range(self.batch_size):
                    if(self.loss_fn=="CTCPSYCLIP"):
                        if(pl[ii][1] > regularizer[ii]):
                            print("too high")
                            print(pl[ii], regularizer[ii])
                            d[ii] = sess.run(self.clip(d[ii], self.psyTh[ii], regularizer[ii], self.psdMaxes[ii], self.max_audio_len, self.window_size, self.step_per_window))
                            sess.run(self.delta.assign(d))
                        else:
                            if(res[ii] == "".join([toks[x] for x in target[ii]])):
                                print("correct")
                                count = 0
                                if(pl[ii] < bestPSY[ii]):
                                    bestCTC[ii] = loss[ii]
                                    bestPSY[ii] = pl[ii]
                                    final_deltas[ii] = new_input[ii]
                                    name = "adv" + str(ii) + "reg" + str(regularizer.reshape(-1)[ii])
                                    if self.mp3: 
                                        convert_mp3(new_input[ii], lengths[ii], name + '.mp3')
                                    wav.write(
                                        "/tmp/" + name + ".wav", 
                                        16000,
                                        np.array(np.clip(np.round(new_input[ii][:lengths[ii]]),
                                                       -2**15, 2**15-1),dtype=np.int16))
                                regularizer[ii] = regularizer[ii]-1
                                sess.run(self.regularizer.assign(regularizer))
                            else:
                                print(res[ii])
                                print("".join([toks[x] for x in target[ii]]))
                    if(res[ii] == "".join([toks[x] for x in target[ii]])):                        
                        if (self.loss_fn == "CTC"):
                            # Get the current constant
                            rescale = sess.run(self.rescale)
                            if rescale[ii]*2000 > np.max(np.abs(d)):
                                # If we're already below the threshold, then
                                # just reduce the threshold to the current
                                # point and save some time.
                                print("It's way over", np.max(np.abs(d[ii]))/2000.0)
                                rescale[ii] = np.max(np.abs(d[ii]))/2000.0

                            # Otherwise reduce it by some constant. The closer
                            # this number is to 1, the better quality the result
                            # will be. The smaller, the quicker we'll converge
                            # on a result but it will be lower quality.
                            rescale[ii] *= .5

                            # Adjust the best solution found so far
                            final_deltas[ii] = new_input[ii]
                            bestCTC = loss
                            bestPSY[ii] = pl[ii]

                            print("Worked i=%d ctcLoss=%f bound=%f"%(ii,loss[ii], 2000*rescale[ii][0]))
                            #print('delta',np.max(np.abs(new_input[ii]-audio[ii])))
                            sess.run(self.rescale.assign(rescale))

                            # Just for debugging, save the adversarial example
                            # to /tmp so we can see it if we want
                            wav.write("/tmp/adv.wav", 16000,
                                      np.array(np.clip(np.round(new_input[ii]),
                                                   -2**15, 2**15-1),dtype=np.int16))
                    
                if(i == MAX-1 and final_deltas[ii] is None):
                    rescale = sess.run(self.rescale)
                    final_deltas[ii] = new_input[ii]
                    print("Did not work i=%d ctcLoss=%f bound=%f"%(ii,loss[ii], 2000*rescale[ii][0]))
                    wav.write("/tmp/adv.wav", 16000,
                              np.array(np.clip(np.round(new_input[ii]),
                                               -2**15, 2**15-1),dtype=np.int16))
            if(b): break
        
        print('bestCTC', bestCTC)
        print('bestPSY', bestPSY)
        if(self.loss_fn=='CTCPSYCLIP'):
            print('regularizer', regularizer.reshape(-1))
        else:
            print('regularizer', 2000*rescale.reshape(-1))
        return final_deltas

def numpyPSD(audio, window_size=2048, step_per_window=4, dB=96):
    frame_length = int(window_size)
    frame_step = int(window_size//step_per_window)
    n_fft = int(2**np.ceil(np.log2(frame_length)))
    win = np.sqrt(8.0 / 3.) * librosa.core.stft(
        y=audio, 
        n_fft=n_fft, 
        hop_length=frame_step, 
        win_length=frame_length, 
        center=False,
        pad_mode='constant')
    z = abs(win / window_size)
    psd = 10 * np.log10(z * z + 0.0000000000000000001)
    psd_max = np.max(psd)
    PSD = 96 - psd_max + psd
    return PSD, psd_max

def log10(x):
  numerator = tf.log(x)
  denominator = tf.log(tf.constant(10, dtype=numerator.dtype))
  return numerator / denominator

def tfPSD(x, window_size=2048, step_per_window=4, psdMax=None):
    batch_size = x.get_shape()[0]
    scale = tf.sqrt(8. / 3.)
    frame_length = int(window_size)
    frame_step = int(window_size//step_per_window)
    fft_length = int(2**np.ceil(np.log2(frame_length)))
    win = tf.scalar_mul(scale, tf.abs(tf.contrib.signal.stft(
        signals=x,
        frame_length=frame_length,
        frame_step=frame_step,
        fft_length=fft_length
        )))
    z = win / frame_length
    psd = tf.reshape(tf.scalar_mul(10, log10((z * z) + 0.0000000000000000001)), [batch_size, -1, window_size//2+1])
    if(len(x.get_shape()) == 2):
        PSD = 96 - tf.reshape(psdMax, [batch_size, 1, 1]) + psd
    else:
        PSD = 96 - tf.reshape(psdMax, [1, 1]) + psd
    return PSD
    
def getMax(tensor):
    tensor=tensor[0]
    ind1 = tf.argmax(tensor)
    temp1 = tf.reduce_max(tensor, 0)
    ind2 = tf.argmax(temp1)
    ind1 = ind1[ind2]
    return tensor[ind1, ind2], ind1, ind2
    
def main(args, thisId):
    print(thisId)
    print(args)
    print("total masking")
    
    with tf.Session() as sess:
        finetune = []
        audios = []
        lengths = []
        psyThs = []
        psdMaxes = []

        f = open(args.input, 'r')
        temp = f.readlines()
        temp = [row[:-1] for row in temp]
        temp = [row.split(",") for row in temp]
        inputFiles = temp[0]
        srcText = temp[1]
        dstText = temp[2]
        outputFiles = [fileName[0] + "_" + thisId + "_out." + fileName[1] for fileName in [fileName.split('.') for fileName in inputFiles]]
        f.close()

        assert len(inputFiles) == len(srcText)
        assert len(srcText) == len(dstText)
        assert len(dstText) == len(outputFiles)
        if args.finetune is not None and len(args.finetune):
            assert len(args.input) == len(args.finetune)
        window_size = 2**8
        step_per_window = 2
        print("window_size, step_per_window", window_size, step_per_window)
        # Load the inputs that we're given
        for i in range(len(inputFiles)):
            fs, audio = wav.read(inputFiles[i])
            if args.mp3:
                audio = convert_mp3(audio, len(audio))[0]
            assert fs == 16000
            assert audio.dtype == np.int16
            print('source dB', 20*np.log10(np.max(np.abs(audio))))
            audios.append(list(audio))
            lengths.append(len(audio))
            if args.finetune is not None:
                finetune.append(list(wav.read(args.finetune[i])[1]))

        maxlen = max(map(len,audios))
        print("maxlen", maxlen)
        audios = np.array([x+[0]*(maxlen-len(x)) for x in audios])
        finetune = np.array([x+[0]*(maxlen-len(x)) for x in finetune])
        
        for audio in audios:
            PSD, psdMax = numpyPSD(audio.astype(float), window_size, step_per_window)
            psdMaxes.append(psdMax)
            frequency = librosa.core.fft_frequencies(fs, int(2**np.ceil(np.log2(window_size))))
            resFreq = frequency[-1] / (frequency.shape[0]-1)
            resTime = window_size/step_per_window/(fs/1000)
            psyTh = reweber.totalMask(PSD, resFreq, resTime, frequency[0], frequency[-1])
            psyTh = psyTh.transpose()
            psyThs.append(psyTh)


        phrase = dstText
        phrase = [[toks.index(c) for c in ph] for ph in phrase]
        maxPhraseLen = np.array([len(p) for p in phrase]).max()

        delta = args.delta
        if(delta != None):
            delta = np.array([wav.read(delta)[1]])
        
        batch_size=len(audios)
        attack = Attack(sess, args.method, maxPhraseLen, maxlen,
                        batch_size=batch_size,
                        mp3=args.mp3,
                        learning_rate=args.lr,
                        onlyCTC = True,
                        window_size=window_size,
                        step_per_window=step_per_window,
                        audio=audios,
                        psdMaxes=np.array(psdMaxes),
                        psdShape=psyThs[0].shape,
                        num_iterations=args.iterations
                        )
        
        deltas = attack.attack(audios,
                               psyThs,
                               lengths,
                               np.array(phrase),
                               regularizer=np.array([args.regularizer]*batch_size).reshape((batch_size)),
                               finetune=finetune)

        # And now save it to the desired output
        if args.mp3:
            for i in range(len(outputFiles)):
                path = outputFiles[i]
                path = path[:path.rfind('.')]+'.mp3'
                print(path)
                filename = path[path.rfind('/')+1:]
                print(filename)
                convert_mp3(deltas[i], lengths[i], filename)
                copyfile("/tmp/" + filename, path)
                print("Final distortion", np.max(np.abs(deltas[0][:lengths[0]]-audios[0][:lengths[0]])))
        else:
            for i in range(len(outputFiles)):
                path = outputFiles[i]
                    
                wav.write(path, 16000,
                          np.array(np.clip(np.round(deltas[i][:lengths[i]]),
                                           -2**15, 2**15-1),dtype=np.int16))
                print("Final distortion", np.max(np.abs(deltas[i][:lengths[i]]-audios[i][:lengths[i]])))



if __name__ == '__main__':
    """
    Do the attack here.

    This is all just boilerplate; nothing interesting
    happens in this method.

    For now we only support using CTC loss and only generating
    one adversarial example at a time.
    """
    parser = argparse.ArgumentParser(description=None)
    
    parser.add_argument('--in', type=str, dest="input", 
                        required=True,
                        help="Input file that defines, input file, output file, sourceText, destText")
    parser.add_argument('--regularizer', type=str,
                        required=False, default=0,
                        help="Initial regularizer")
    parser.add_argument('--lr', type=int,
                        required=False, default=100,
                        help="Learning rate for optimization")
    parser.add_argument('--iterations', type=int,
                        required=False, default=2000,
                        help="Maximum number of iterations of gradient descent")
    parser.add_argument('--l2penalty', type=float,
                        required=False, default=float('inf'),
                        help="Weight for l2 penalty on loss function")
    parser.add_argument('--mp3', action="store_const", const=True,
                        required=False,
                        help="Generate MP3 compression resistant adversarial examples")
    parser.add_argument('--method', type=str, 
                        required=False, default='CTCPSYCLIP',
                        help="CTC, CTCPSYCLIP are availabel")
    args = parser.parse_args()
    print(args)
    toLog = True
    thisId = str(int(round(time.time())))
    if(toLog):
        orig_stdout = sys.stdout
        f = open('log_' + args.method + '_' + thisId + '.txt', 'w')
        sys.stdout = f
        main(args, thisId)
        sys.stdout = orig_stdout
        f.close()
    else: main(args, thisId)

# relu instead of softplus
#main()
