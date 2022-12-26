# handGesture Recognition and Detection
our project contains 5 parts that 3 parts are Implementation of CNN network and 2 parts are Implementation of detection hand gestures:

1. say numbers

2. say i love you

3. say fist,palm,swing

4. say direction of mouse

5. volume control


**Requirements:**

```
pip install -r requirements.txt
```


## How to run

if you want to use pretrained models, use command below.

```
python hand_gesture_main_program.py
```

but if you want to train again the models follow steps below:

for training i_love_you Run :
```
python Recognize_Gesture.py
```
for training say_numbers Run:
```
python traingest.py
```
for training posenet Run:
```
python ModelTrainer.py
```
