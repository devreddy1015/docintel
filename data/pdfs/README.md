# Put tender PDFs here

One folder per tender, named with its GePNIC Tender ID:

```
data/pdfs/2026_AMU_926330_3/Tendernotice_1.pdf
data/pdfs/2026_AMU_926330_3/BOQ.pdf
data/pdfs/2026_DDA_928692_1/NIT.pdf
```

(Or put the Tender ID in the file name: `2026_AMU_926330_3_NIT.pdf`.)

Download the documents from the tender's page on https://eprocure.gov.in/eprocure/app
(the portal asks you to solve a CAPTCHA, which is why this step is manual), then run
`make ingest`. Re-running is safe: files are identified by their SHA-256, so nothing is
ingested twice.

`synthetic/` holds generated test notices (every page is marked SYNTHETIC TEST DOCUMENT).
