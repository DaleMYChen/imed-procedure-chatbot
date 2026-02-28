### Action step 1. 
```
Automatic reply: Patient query — MRI cost enquiry
I-MED Accounts <accounts@i-med.com.au>	Fri, Feb 27, 2026 at 4:43 PM
To: M Chen <chenmuye5230@gmail.com>
Thank you for reaching out to the I-MED Accounts Receivable team.
For Our Patients:

We’re here to help with your invoice enquiry.
If your enquiry is urgent, especially regarding an invoice, please call us directly on 03 8587 5344 (let the operator know you’ve also sent an email so we can remove it from the inbox).
If your enquiry is regarding a quote, booking, or requesting images or scans:
Please note that our Accounts Receivable team are unable to assist with appointment bookings or quotes, or access to images or scan results.

To ensure you receive the fastest service, we kindly ask you to:
Book online: Book an Appointment
Or to find a clinic near you for quotes, enquiries or image requests: Find a Radiology Clinic

For Solicitor Correspondence, Court Ordered Documents or Requests for Medical Records
 
If your enquiry relates to solicitor correspondence, court-ordered documents (including subpoenas) or requests for medical records, please be aware that this inbox is not monitored for legal matters.
 
Due to the high volume of emails received, it is your responsibility to forward such requests directly to the Privacy Team at medicalrecordsrequest@i-med.com.au. This inbox will not forward legal correspondence on your behalf, and failure to direct your request correctly may result in delays or non-compliance with timeframes.
 
 
For our Corporate and Government Customers (not for individual patient enquiries):

If you are contacting us on behalf of a business, government department, insurer, or legal entity, please use the following direct contacts in future to ensure a quicker response (we will transfer your email to the correct department in this instance – no need to resend):
Remittance advices:
Remittances@i-med.com.au
WorkCover, Motor Vehicle Accident, or Australian Defence Force enquiries:
WC-MVA-ADF-Support@i-med.com.au
Private Health Funds, Department of Veteran Affairs, or Medicare enquiries:
Private.healthfunds@i-med.com.au
All other Corporate or Government organisations:
hospital.contracts@i-med.com.au
Thank you for your cooperation. 

Kind regards,
I-MED Accounts Receivable Team
```

Topics of costs and Medicare rebates can lead to personally identifiable information shared in the subsequent conversation. In the medical service domain we can not expose the chatbot LLM to these PII information due to privacy concerns and policies. A possible situation can be the chatbot asking the patient for his Medicare number or medical history to determine eligibility of a service. The current implementation has no secure data handling infrastructure: no authentication or encrypted storage. 

Therefore the chatbot is instructed to direct such inquiries to human staff. Data available to the chatbot is limited to the general information on services & procedures. 

---

### Action step 2. 
```
    "title": "Find a Clinic",
    "url": "https://i-med.com.au/find-a-radiology-clinic",
    "raw_text": "Our network of radiology clinics\nPostcode or suburb Procedure",
```
The "Find a Clinic" page is just an empty form waiting for user input, in order to load relevant information. No static information on postcode or suburb to scrap.  

---

### Q1. 

#### Query 1. Two questions from the same page. 
```
"question": "How long will a PET scan take me, and how to prepare for it?"
```
Response:
```
{
  "answer": "A PET scan typically takes between two to three hours, including preparation time. To prepare for a PET scan:\n\n* Arrive at the clinic\n* Change into a gown\n* A small needle will be inserted into a vein in your arm or the back of your hand to fit an intravenous line\n* A brief medical history will be taken and your blood sugar level checked\n* The radioactive substance is injected into your vein through the intravenous line\n* If you are having an FDG PET scan, rest quietly for 90 minutes without movement or talking\n* You may be asked to drink some contrast material that moves through your stomach and bowel\n* A catheter (if necessary) will be placed into your bladder to improve image quality\n\nThe actual PET scan time is approximately 30 minutes, but this can vary depending on the regions of your body being scanned.",
  "sources": [
    {
      "title": "PET Scan",
      "url": "https://i-med.com.au/procedures/pet-scan",
      "section": "How long does a PET scan take? keyboard_arrow_down"
    }
  ],
  "error": null
}
```

#### Query 2. Two questions from separate pages.
```
"question": "Am I eligible for a lung screening if i am 48? If i have a history of kidney disease, how shall i prepare for Angiography?"
```
Response:
```
{
  "answer": "For Lung Screening:\nYou are not eligible for the National Lung Cancer Screening Program (NLCSP) at this age. You need to be between 50 and 70 years old.\n\nFor Angiography with a history of kidney disease:\nIf you have a history of kidney disease, you must inform staff when you make the appointment and when you attend for the procedure. This ensures that the appropriate measures are taken to carry out the procedure with maximum safety.",
  "sources": [
    {
      "title": "Lung Screening",
      "url": "https://i-med.com.au/procedures/lung-screening",
      "section": "Who is eligible for the screening program?"
    },
    {
      "title": "Angiography",
      "url": "https://i-med.com.au/procedures/angiography",
      "section": "How do I prepare for angiography? keyboard_arrow_down"
    }
  ],
  "error": null
}
```

##### Chunks and scores
```
Query: 'Am I eligible for a lung screening if i am 48? If i have a history of kidney disease, how shall i prepare for Angiography?'
Chunks retrieved: 3

--- Chunk 1 ---
Procedure : Lung Screening
Section   : Who is eligible for the screening program?
URL       : https://i-med.com.au/procedures/lung-screening
Score     : 0.6671
Text      :
Who is eligible for the screening program?
To be eligible for the National Lung Cancer Screening Program (NLCSP), you need to: • be aged between 50 and 70 years • show no signs or symptoms of lung cancer • have a history of at least 30 pack-years of cigarette smoking • be still smoking or have quit in the past 10 years. If eligible, you will need to see your GP as a first step and obtain an imaging request for a low dose CT scan (CAT scan) of your chest. If you do not know how many ‘pack-years’ of smoking you have, your GP can help work this out.

--- Chunk 2 ---
Procedure : Angiography
Section   : How do I prepare for angiography? keyboard_arrow_down
URL       : https://i-med.com.au/procedures/angiography
Score     : 0.5685
Text      :
How do I prepare for angiography? keyboard_arrow_down
There are general preparations plus others that might be required depending on which part of the body is to be examined. Our staff will advise you on this when you make the booking. General preparation ·     Some procedures require fasting, specific advice will be given to you before your procedure. ·     Bring all of your usual medication(s) (or a list of your medications) with you to your appointment. ·     Bring any relevant previous imaging (angiograms, X-rays, CT scans, ultrasound or MRIs etc.), so that the medical staff can have all relevant information available on the day. ·     It is useful to arrive at the clinic before the scheduled appointment time so that staff can obtain your medical history, collect information about your current medical problem and discuss the procedure with you. ·     You will be asked to sign a consent form indicating that you understand what is going to happen, you are happy for the procedure to be carried out and your questions have been satisfactorily answered. ·     If you have a history of kidney disease, have previously had an allergic reaction to contrast medium, suffer from any other allergies or have diabetes, you must tell staff when you make the appointment and when you attend for the procedure. This ensures that the appropriate measures are taken to carry out the procedure with maximum safety. ·     If you are taking metformin, aspirin, clopidogrel, warfarin or other blood thinning medications, you must inform staff when you make the appointment and when you attend for the procedure. ·     Make arrangements with a relative or friend to drive you home after the procedure, as you might be given medication that will make you drowsy. ·     If you live a significant distance from a hospital, it might be better in the 24 hours after the procedure, for you to stay overnight nearby.

--- Chunk 3 ---
Procedure : Lung Screening
Section   : How do I access lung screening at I-MED?
URL       : https://i-med.com.au/procedures/lung-screening
Score     : 0.4963
Text      :
How do I access lung screening at I-MED?
As the largest network of radiology clinics in Australia, we offer a comprehensive range of scanning services at a clinic near you, including low dose CT for lung screening. • Our expert lung imaging teams provide high quality, low dose CT scans and report back to your doctor within 3 working days. • There are no costs associated with your lung screening scan at I-MED, we provide a free service as part of the government’s bulk billed screening program. • Our staff are trained to answer all your questions about the process and support you every step of the way, from booking online to visiting the clinic for your lung scan through to receiving your results for discussion with your doctor. • You can access our streamlined online booking system , available 24/7, and find the best time and location for your low-dose lung screening.
```


#### Query 3. Additional question out of scrapping scope.
```
"question": "Am I eligible for a lung screening if i am 48? If i have a history of kidney disease, how shall i prepare for Angiography? What is a gamma camera in Nuclear medicine?"
```
Note: Nuclear medicine procedure page is not one of the scrapped pages.

Response:
```
{
  "answer": "I don't have enough information about that in the I-MED procedure content. Please instead submit an enquiry at https://i-med.com.au/contact-us\n\nI don't have enough information about that in the I-MED procedure content. Please instead submit an enquiry at https://i-med.com.au/contact-us\n\nI don't have enough information about gamma cameras in Nuclear medicine within the provided context. Please instead submit an enquiry at https://i-med.com.au/contact-us",
  "sources": [
    {
      "title": "Lung Screening",
      "url": "https://i-med.com.au/procedures/lung-screening",
      "section": "Who is eligible for the screening program?"
    },
    {
      "title": "Angiography",
      "url": "https://i-med.com.au/procedures/angiography",
      "section": "How do I prepare for angiography? keyboard_arrow_down"
    }
  ],
  "error": null
}
```

---

### Q2. 
Refer to query 3 and the corresponding response in Q1: the same top-3 chunks were retrieved. Ideal output for query 3: answering the first 2 subparts with relevant information (as shown in query 2 of Q1), then tell the user the 3rd part of the question is out of scope. 

Behaviour: The LLM repeated the 'out-of-scope' response 3 times, meaning that it recognised the query had 3 distinct sub-questions and structured its response with 3 corresponding parts. But the chatbot avoided answering any part of the query even though the first 2 parts were relevant and had chunks already retrieved.

The root cause lies in prompt building: all three sub-questions were passed as a single string sharing one context block, with no instruction to process and reason each one independently. 

Relevant code is in `llm._build_prompt()`:
```
return f"""...
    Context:
    {context}        

    Question: {question}  

    Answer:"""
```

When the LLM encountered one unanswerable sub-question (gamma camera from a page not scraped), it applied its fallback instruction globally and returned "no information" for all three parts despite having sufficient context for the other parts. 

A production fix would decompose compound queries into individual questions before retrieval, and run separate RAG passes. It will ensure each sub-question gets its own context budget and independent LLM reasoning.

---

### Q3. 

1. Asking for clinic details. 

Gap: data exists, but scrapper can't access. 

A patient might ask for the nearest I-MED clinic to his suburb and the contact number. But he would not be able to obtain an answer because the current implementation of the scrapper only has access to statis contents. In `/find-a-radiology-clinic` he will have to input the location for the relevant information to be loaded. 

2. Asking for scan results. 

Gap: data does not exist on website.

A patient asking for details of a recent scan will not receive relevant information from the chatbot, because personally identifiable information of the patients is not accessible by the chatbot. The chatbot will instead direct the patient to contact the internal staff for his records. 

3. Scan preparation instructions. 

Gap: data exists but the chatbot is not flexible with the language style of query. 

A patient asking for the details of scan preparation may receive a passive response depending how he phrases the question: 
```
"question": "i have been taking warfarin, can i still do an angiography scan?"
"answer": "I don't have enough information about that in the I-MED procedure content. Please instead submit an enquiry at https://i-med.com.au/contact-us",
"section": "What is angiography?"
```
The question tone was casual and the chatbot based the response on an incorrect section. 

If the question involved keyword 'prepare', the chatbot would identify the correct section and output an informative response:
```
"question": "i have been taking warfarin, how do i prepare for an angiography scan?"
"answer":"You will need to inform staff when you make the appointment and when you attend for the procedure that you are taking warfarin. This ensures that the appropriate measures are taken to carry out the procedure with maximum safety."
"section": "How do I prepare for angiography? keyboard_arrow_down"
```

---

### Q4. 

All 3 prep types fell under the same chunk (Chunk 1), because the chunking strategy placed the entire body text in a single chunk. 

Top-3 chunks for CT scan page:
```
Query: 'Do I need to fast before a chest CT scan?'
Chunks retrieved: 3

--- Chunk 1 ---
Procedure : CT Scan
Section   : How do I prepare for a CT scan? keyboard_arrow_down
URL       : https://i-med.com.au/procedures/ct-scan
Score     : 0.7636
Text      :
How do I prepare for a CT scan? keyboard_arrow_down
If necessary, our clinic will provide you with instructions for your CT scan prior to your appointment. These instructions are very important as they may affect the accuracy of the test or require that the test be rebooked if you are not properly prepared for the CT scan. Some tests require no preparation, these include: brain, sinus or facial bones, temporal bones (inner ear), spine, knee or wrist, and CT scans of the bones. Many types of CT scan require an injection of an iodinated contrast material to show blood vessels and some organs. For these tests we will ask you to fast (not eat) prior to your appointment. It is important that the need to fast does not affect you if you have special dietary requirements (e.g. diabetes). Please check with your doctor or our clinic if you have any concerns. Chest CT preparation: • Two hour fast • Medication as normal Abdomen/pelvis CT & abdomen/pelvis + chest CT: • Two hour fast • Drink 1 litre of water, one hour before your appointment time • Can go to the bathroom, no need to hold bladder • Medication as normal All patients are asked if they have kidney disease, diabetes or if taking metformin. If ‘yes’ to any of the above, ask patient to bring copies of recent blood tests (within the last 3 months) to their appointment. If you do require an iodinated contrast injection for your test, the radiologist or radiographer will discuss this with you. They will then use a needle to insert a cannula (a small plastic tube) into a vein in your arm or the back of your hand so that the iodine contrast can be inserted into the cannula during the test. While the iodinated contrast used for injections is considered very safe, there are precautions that must be taken when using it, particularly if you have poor kidney function or diabetes. Tests investigating your abdomen may require you to drink a different kind of iodinated contrast solution to outline your intestine (part of your digestive system). This will also require fasting. This drink is given in a different way depending where you are having the CT scan done. You will usually be asked to drink part of the whole dose an hour prior to the scanning time and the rest of it just before entering the scanning room. Depending on the type of scan that you are having you may be asked to change into a gown to avoid parts of your clothing affecting the scan. If you have any concerns regarding fasting, the iodinated contrast injection or your medication you should contact your own doctor or our clinic prior to your appointment. It is important to follow the instructions you are given to ensure that the test is done safely, accurately and efficiently and so that you do not need to have the scan rescheduled or repeated.

--- Chunk 2 ---
Procedure : CT Scan
Section   : How long does a CT scan take? keyboard_arrow_down
URL       : https://i-med.com.au/procedures/ct-scan
Score     : 0.5796
Text      :
How long does a CT scan take? keyboard_arrow_down
The time taken to complete the CT scan will vary, depending on the examination that has been requested by your referring doctor. CT scans that do not require an injection and are usually quite quick and may be completed within 5 minutes. In the case of CT scans which require you to drink a contrast solution or have an injection, the preparation time is often much longer than that of the scan itself. When a drink is required for an abdominal scan (of your stomach), you are often asked to have that drink an hour before the appointment time. This may be done prior to you arriving at the hospital or radiology practice or while you are in the waiting area. Even when you are having a CT scan that requires an injection or a drink and other preparation, the time taken for the scan is usually under ten minutes.

--- Chunk 3 ---
Procedure : CT Scan
Section   : What happens during a CT scan? keyboard_arrow_down
URL       : https://i-med.com.au/procedures/ct-scan
Score     : 0.5374
Text      :
What happens during a CT scan? keyboard_arrow_down
CT scans are designed to look at specific parts of the body and are tailored for each person, to investigate their particular condition. This means that all CT scans are slightly different. The CT scan equipment is a large square machine with a circular hole, sometimes described as looking like a “donut”. The process involves you lying on a bed attached to the scanner (this may be feet first or head first depending on the part of the body being looked at). The bed will then be raised up to a height level with the circular hole in the scanner and the bed slides in and out of the hole several times while pictures are being taken. It is important to try not to move during the scan as it will affect the quality of the pictures and make them harder for the radiologist to interpret. The radiographer performing the CT scan may ask you to hold your breath for some scans. The length of time for each breath hold is usually under 10 seconds. Most scanners in use now are able to give instructions in different languages to help you understand what you need to do and what is happening. They will also often have ways of communicating with you if your hearing is poor. The first few scans are usually done to set up the machine ready for the test. When the test is programmed into the computer by the radiographer and the scan is ready to go, they may remind you to keep still. If your test requires an iodinated contrast injection, the radiographer will come into the room to administer it using either a hand held syringe or a mechanical pump. The pump helps to put the iodinated contrast in at a set rate and allows for the scanner to target specific areas of the body. When the iodinated contrast is injected, most people will get a strange metallic taste in the mouth and feel a warm sensation through the body. This warm sensation may concentrate around the groin or buttock region and can feel like you may have wet yourself, even though you have not. Do not be concerned if this happens, it is a common sensation and usually goes away within a couple of minutes. Once the radiographer has reviewed the images briefly to check that the appropriate areas have been shown, they will come into the room to help you off the bed. The radiographer will not be able to give you any results after the CT scan; this is the responsibility of your doctor and the radiologist who interprets the images from the scan and provides a report to your doctor. Once the CT scan is completed, you will have the cannula removed so you can go home.
```



The chatbot response was actually correct. Llama was able to find the relevant information about chest CT within the large chunk:
```
{
  "answer": "Yes, for a chest CT scan, you will be asked to fast (not eat) for two hours prior to your appointment.",
  "sources": [
    {
      "title": "CT Scan",
      "url": "https://i-med.com.au/procedures/ct-scan",
      "section": "How do I prepare for a CT scan? keyboard_arrow_down"
    },
    {
      "title": "Lung Screening",
      "url": "https://i-med.com.au/procedures/lung-screening",
      "section": "What does the lung screening scan involve?"
    }
  ],
  "error": null
}
```

However if the question is more subtle, this rigid chunking strategy will leave the entire interpretation workload to the LLM at generation time. This is a known limitation of heading-based chunking: retrieval granularity is determined by how the source page is structured, not by what would be most useful for the user.

A semantic chunking strategy is required to create context-aware chunks, and to improve retrieval quality. 