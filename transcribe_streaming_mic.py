#!/usr/bin/env python

from __future__ import division
import re
import sys
import os
os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="/home/intuitivecompting/Downloads/Speech to Text-888f6a05acbb.json"
from google.cloud import speech
from google.cloud.speech import enums
from google.cloud.speech import types
import pyaudio
from six.moves import queue
import datetime
import rospy
from std_msgs.msg import Int32
from std_msgs.msg import String

# [END import_libraries]

# Audio recording parameters
RATE = 16000
CHUNK = int(RATE / 10)  # 100ms
feedback = 0
verbals = r'\b(attach|move|make|moves|who|attached|attack|attacked|moved|get|grab|take|what|movie|movies|pic|pics|pick|select|put|transport)\b'
objects = r'\b(object|item|objects|items|one|ones|cube|blocks|block|rick|rock|guys|guy|gay|gays|target|targets|it)\b'
places = r'\b(here|there|shear|Kia|cheer|place|location|place|hear|hair|position|gear|sure)\b'
adjs = r'\b(yellow|green|blue|small|big|two|three)\b'
pointings = r'\b(this|that|these|those|lease)\b'
quiting = r'\b(exit|quit)\b'


ready_for_pick = False
ready_for_place = False
ready_for_color = False

class MicrophoneStream(object):
    """Opens a recording stream as a generator yielding the audio chunks."""
    def __init__(self, rate, chunk):
        self._rate = rate
        self._chunk = chunk

        # Create a thread-safe buffer of audio data
        self._buff = queue.Queue()
        self.closed = True

    def __enter__(self):
        self._audio_interface = pyaudio.PyAudio()
        self._audio_stream = self._audio_interface.open(
            format=pyaudio.paInt16,
            # The API currently only supports 1-channel (mono) audio
            # https://goo.gl/z757pE
            channels=1, rate=self._rate,
            input=True, frames_per_buffer=self._chunk,
            # Run the audio stream asynchronously to fill the buffer object.
            # This is necessary so that the input device's buffer doesn't
            # overflow while the calling thread makes network requests, etc.
            stream_callback=self._fill_buffer,
        )

        self.closed = False

        return self

    def __exit__(self, type, value, traceback):
        self._audio_stream.stop_stream()
        self._audio_stream.close()
        self.closed = True
        # Signal the generator to terminate so that the client's
        # streaming_recognize method will not block the process termination.
        self._buff.put(None)
        self._audio_interface.terminate()

    def _fill_buffer(self, in_data, frame_count, time_info, status_flags):
        """Continuously collect data from the audio stream, into the buffer."""
        self._buff.put(in_data)
        return None, pyaudio.paContinue

    def generator(self):
        while not self.closed:
            # Use a blocking get() to ensure there's at least one chunk of
            # data, and stop iteration if the chunk is None, indicating the
            # end of the audio stream.
            chunk = self._buff.get()
            if chunk is None:
                return
            data = [chunk]

            # Now consume whatever other data's still buffered.
            while True:
                try:
                    chunk = self._buff.get(block=False)
                    if chunk is None:
                        return
                    data.append(chunk)
                except queue.Empty:
                    break

            yield b''.join(data)
# [END audio_stream]
def detect_verb(transcript):
    global verbals
    global ready_for_pick
    global ready_for_place
    global ready_for_color
    global feedback
    if feedback == 1:
        ready_for_pick = False
        ready_for_place = False
        feedback = 0
    elif feedback == 2:
        ready_for_place = False
        ready_for_pick = True
        feedback = 0
    if re.search(verbals, transcript, re.I):
        return True
    else:
        return False

def detect_place(transcript):
    global places
    global ready_for_pick
    global ready_for_place
    global feedback
    if feedback == 1:
        ready_for_pick = False
        ready_for_place = False
        feedback = 0
    elif feedback == 2:
        ready_for_place = False
        ready_for_pick = True
        feedback = 0
    if re.search(places, transcript, re.I):
        return True
    else:
        return False

def detect_quit(transcript):
    global quiting
    global ready_for_pick
    global ready_for_place
    global feedback
    if feedback == 1:
        ready_for_pick = False
        ready_for_place = False
        feedback = 0
    elif feedback == 2:
        ready_for_place = False
        ready_for_pick = True
        feedback = 0
    if re.search(quiting, transcript, re.I):
        return True
    else:
        return False


def detect_object(transcript):
    global objects
    global ready_for_pick
    global ready_for_place
    global feedback
    if feedback == 1:
        ready_for_pick = False
        ready_for_place = False
        feedback = 0
    elif feedback == 2:
        ready_for_place = False
        ready_for_pick = True
        feedback = 0
    if re.search(objects, transcript, re.I):
        return True
    else:
        return False


def detect_adj(transcript):
    global adjs
    global ready_for_pick
    global ready_for_place
    global feedback
    if feedback == 1:
        ready_for_pick = False
        ready_for_place = False
        feedback = 0
    elif feedback == 2:
        ready_for_place = False
        ready_for_pick = True
        feedback = 0
    if re.search(adjs, transcript, re.I):
        if re.search(r'\b(blue)\b', transcript, re.I):
            return "blue"
            # pub_color.publish("blue")
        elif re.search(r'\b(yellow)\b', transcript, re.I):
            return "yellow"
            # pub_color.publish("yellow")
        elif re.search(r'\b(green)\b', transcript, re.I):
            return "green"
            # pub_color.publish("green")
    else:
        return None


def detect_pointing(transcript):
    global pointings
    global ready_for_pick
    global ready_for_place
    global feedback
    if feedback == 1:
        ready_for_pick = False
        ready_for_place = False
        feedback = 0
    elif feedback == 2:
        ready_for_place = False
        ready_for_pick = True
        feedback = 0
    if re.search(pointings, transcript, re.I):
        return True
    else:
        return False

def callback(msg):
    global feedback
    feedback = msg.data


def listen_print_loop(responses):
    global ready_for_place
    global ready_for_pick
    global ready_for_color
    num_chars_printed = 0
    pub_select = rospy.Publisher('/voice_command', Int32, queue_size=1)
    pub_color = rospy.Publisher('/item_color', String, queue_size=20)
    #sub = rospy.Subscriber('/feedback', Int32, callback)
    pick_time = None
    start_time = datetime.datetime.now()
    for response in responses:


        if not response.results:
            continue
        result = response.results[0]
        #print(result)
        if not result.alternatives:
            continue

        transcript = result.alternatives[0].transcript

        print(transcript)
        if detect_quit(transcript):
            print('Exiting..')
            pub_select.publish(-1)
            return False

        elif detect_verb(transcript):
            if not ready_for_place:
                ready_for_pick = True
                ready_for_color = True
            #pub_select.publish(1)

        if detect_adj(transcript) is not None and ready_for_color:
            pub_color.publish(detect_adj(transcript))
            ready_for_color = False
            print(detect_adj(transcript))

        if (detect_pointing(transcript) or detect_object(transcript)) and ready_for_pick:
            print(1)
            pub_select.publish(1)
            ready_for_pick = False
            ready_for_place = True
            pick_time = datetime.datetime.now()

        if detect_place(transcript) and ready_for_place:
            print(2)
            pub_select.publish(2)
            ready_for_place = False
            ready_for_pick = False
            ready_for_color = False
            return True

        current_time = datetime.datetime.now()
        if (current_time - start_time).seconds > 60:
            return True

        if pick_time is not None:
            if (current_time - pick_time).seconds > 5:
                print(-20)
                ready_for_place = False
                ready_for_pick = False
                pub_select.publish(-20)
                pick_time = None

        if re.search(r'\b(cancel|pencil)\b', transcript, re.I):
            print(9)
            ready_for_place = False
            ready_for_pick = False
            pub_select.publish(9)
            pick_time = None

        if re.search(r'\b(stop)\b', transcript, re.I):
            print(-9)
            ready_for_place = False
            ready_for_pick = False
            pub_select.publish(-9)
            pick_time = None


def main():
    rospy.init_node('Speech_node')
    language_code = 'en-US'
    if_restart = True
    client = speech.SpeechClient()
    config = types.RecognitionConfig(
        encoding=enums.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=RATE,
        language_code=language_code,
        speech_contexts = [{"phrases":["attach","move","make","get","grab","take","pick","select","put","object","item","objects",
                                       "items","one","ones","cube","cubes","blocks","block","guys","guy","here","there","place",
                                       "location","position","yellow","green","blue","this","that","these","those","targets","target",
                                       "it", "transport"]}])
    streaming_config = types.StreamingRecognitionConfig(
        config=config,
        interim_results=True)
    while if_restart:
        with MicrophoneStream(RATE, CHUNK) as stream:
            audio_generator = stream.generator()
            requests = (types.StreamingRecognizeRequest(audio_content=content)
                        for content in audio_generator)

            responses = client.streaming_recognize(streaming_config, requests)
            if_restart = listen_print_loop(responses)




if __name__ == '__main__':
    main()

