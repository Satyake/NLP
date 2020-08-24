import nltk 
nltk.download()
nltk.download('punkt')
paragraph="""In 1981, the U.S. Air Force identified a requirement for an Advanced Tactical Fighter (ATF) to replace the F-15 Eagle and F-16 Fighting Falcon. Code named "Senior Sky", this air-superiority fighter program was influenced by emerging worldwide threats, including new developments in Soviet air defense systems and the proliferation of the Su-27 "Flanker"- and MiG-29 "Fulcrum"-class of fighter aircraft.[8] It would take advantage of the new technologies in fighter design on the horizon, including composite materials, lightweight alloys, advanced flight control systems, more powerful propulsion systems, and most importantly, stealth technology. In 1983, the ATF concept development team became the System Program Office (SPO) and managed the program at Wright-Patterson Air Force Base. The demonstration and validation (Dem/Val) request for proposals (RFP) was issued in September 1985, with requirements placing strong emphasis on stealth and supercruise. Of the seven bidding companies, Lockheed and Northrop were selected on 31 October 1986. Lockheed teamed with Boeing and General Dynamics while Northrop teamed with McDonnell Douglas, and the two contractor teams undertook a 50-month Dem/Val phase, culminating in the flight test of two technology demonstrator prototypes, the YF-22 and the YF-23, respectively. Concurrently, Pratt & Whitney and General Electric were awarded contracts to develop the YF119 and YF120 respectively for the ATF engine competition.[9][10]

Dem/Val was focused on risk reduction and technology development plans over specific aircraft designs.[N 2] Contractors made extensive use of analytical and empirical methods, including computational fluid dynamics, wind-tunnel testing, and radar cross-section calculations and pole testing; the Lockheed team would conduct nearly 18,000 hours of wind-tunnel testing. Avionics development was marked by extensive testing and prototyping and supported by ground and flying laboratories.[12] During Dem/Val, the SPO used the results of performance and cost trade studies conducted by contractor teams to adjust ATF requirements and delete ones that were significant weight and cost drivers while having marginal value. The short takeoff and landing (STOL) requirement was relaxed in order to delete thrust-reversers, saving substantial weight. As avionics was a major cost driver, side-looking radars were deleted, and the dedicated infra-red search and track (IRST) system was downgraded from multi-color to single color and then deleted as well. However, space and cooling provisions were retained to allow for future addition of these components. The ejection seat requirement was downgraded from a fresh design to the existing McDonnell Douglas ACES II. Despite efforts by the contractor teams to rein in weight, the takeoff gross weight estimate was increased from 50,000 lb (22,700 kg) to 60,000 lb (27,200 kg), resulting in engine thrust requirement increasing from 30,000 lbf (133 kN) to 35,000 lbf (156 kN) class.[13]

Each team produced two prototype air vehicles for Dem/Val, one for each of the two engine options. The YF-22 had its maiden flight on 29 September 1990 and in flight tests achieved up to Mach 1.58 in supercruise. After the Dem/Val flight test of the prototypes, on 23 April 1991, Secretary of the USAF Donald Rice announced the Lockheed team and Pratt & Whitney as the winners of the ATF and engine competitions.[14] The YF-23 design was considered stealthier and faster, while the YF-22, with its thrust vectoring nozzles, was more maneuverable as well as less expensive and risky.[15] The aviation press speculated that the Lockheed team's design was also more adaptable to the U.S. Navy's Navalized Advanced Tactical Fighter (NATF),[N 3] but by 1991, the Navy had abandoned NATF"""

import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer

ps=PorterStemmer()
lm=WordNetLemmatizer()

sentences=nltk.sent_tokenize(paragraph)

sentences=[nltk.word_tokenize(word) for word in sentences]

for i in range(len(sentences)):
    sentences[i]=[word for word in sentences[i] if word not in stopwords.words('english')]
from sklearn.feature_extraction.text import CountVectorizer
CV=CountVectorizer()
sentence1=CV.fit_transform(sentences[0]).toarray()  # use this line to navigate the sentence list
