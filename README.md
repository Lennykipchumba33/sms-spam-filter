# SMS Spam Filter

A real-world text-classification project using the **UCI SMS Spam Collection**, a labeled corpus of 5,574 English SMS messages.

## Data source and attribution

Download the official archive from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/228/sms+spam+collection) or its [official data URL](https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip).

The collection combines messages from several documented sources, including the UK Grumbletext forum and the NUS SMS Corpus. UCI lists the dataset under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/legalcode). The archive readme also requests citation of Almeida and Hidalgo (2011) and includes its own as-is/no-warranty notice. Retain this attribution when redistributing.

Citation: Almeida, T. & Hidalgo, J. (2011). SMS Spam Collection. UCI Machine Learning Repository. https://doi.org/10.24432/C5CC84

## Run

```bash
pip install -r requirements.txt
python -m src.train --data data/SMSSpamCollection
```

The model uses TF-IDF word and character features with logistic regression and reports precision, recall, F1, and a confusion matrix. The split is stratified, but deployment drift is not modeled.

The original code is MIT-licensed; the dataset remains under its source terms.
