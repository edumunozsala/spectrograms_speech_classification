# Using Machine Learning to identify accents in spectrograms of speech

[Blog post](https://https://www.medium.com/p/5db91c191b6b/edit)

## Content
This repository contains some files I build in the development of the Capstone Project in the Microsoft Professional Program in Artificial Intelligence in April - May 2019. 

## Project Description
Voice recognition software enables our devices to be responsive to our speech. We see it in our phones, cars, and home appliances. But for people with accents - even the regional lilts, dialects and drawls native to various parts of the same country - the artificially intelligent speakers can seem very different: inattentive, unresponsive, even isolating. Researchers found that smart speakers made about 30 percent more errors in parsing the speech of non-native speakers compared to native speakers. Other research has shown that voice recognition software often [works better for men than women](https://medium.com/r/?url=https%3A%2F%2Fwww.dailydot.com%2Fdebug%2Fgoogle-voice-recognition-gender-bias%2F).

Algorithmic biases often stem from the datasets on which they're trained. One of the ways to improve non-native speakers' experiences with voice recognition software is to train the algorithms on a diverse set of speech samples. Accent detection of existing speech samples can help with the generation of these training datasets, which is an important step toward closing the "accent gap" and eliminating biases in voice recognition software.

## About the data
A spectrogram is a visual representation of the various frequencies of sound as they vary with time. The x-axis represents time (in seconds), and the y-axis represents frequency (measured in Hz). The colors indicate the amplitude of a particular frequency at a particular time (i.e., how loud it is).
These spectrograms were generated from audio samples in the [Mozilla Common Voice dataset](https://medium.com/r/?url=https%3A%2F%2Fvoice.mozilla.org%2Fen%2Fdatasets). Each speech clip was sampled at 22,050 Hz, and contains an accent from one of the following three countries: Canada, India, and England. For more information on spectrograms, see the [home page](https://medium.com/r/?url=https%3A%2F%2Fdatasciencecapstone.org%2Fcompetitions%2F16%2Fidentifying-accents-speech%2Fpage%2F49%2F).

## Machine learning model based on CNN 
A full description on this algorithm and how it works is detailed in my blog post [Using Machine Learning for identifying accents in spectrograms of speech]((https://https://www.medium.com/p/5db91c191b6b/edit)).

### Show some commands
```javascript
var browserify = require('browserify');
var fs = require('fs');

var b = browserify('main.js');
b.transform('can.viewify');

b.bundle().pipe(fs.createWriteStream('bundle.js'));
```

## Contributing
If you find some bug or typo, please fixit and push it to be applied 

## License

These notebooks are under a public GNU License.