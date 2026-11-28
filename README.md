# Test build: letters-only recitation checker (Qalun / Hafs)
Static web app. Runs fully on the phone. No server, no accounts, no recordings stored.
Checks letters only. Does not check vowels, madd, or the Qalun/Hafs difference.
Model: Abdou/quran-asr-small (MIT). Texts: KFGQPC Qalun v10 and Hafs v18.


## Credits and licenses
- Quran text: King Fahd Glorious Quran Printing Complex (KFGQPC), Qalun v10 and Hafs v18, shown unmodified, used for a free non-commercial test. Rights belong to KFGQPC.
- Recognition model: Abdou/quran-asr-small (MIT), trained in part on tarteel-ai/tlog (CC-BY-4.0).
- Speech runtime: ONNX Runtime (MIT, Microsoft).

## Identity and data sources
- App icon and store banner: designed by the project owner (icon-192.png, icon-512.png, favicon-48.png, banner.jpg).
- Tests page (tests.html): Qalun page starts from the official KFGQPC Qaloon data (King Fahd Complex); Hafs page starts from Tanzil metadata; similar verses (mutashabihat) from github.com/Waqar144/Quran_Mutashabihat_Data (CC0), based on the work of Qari Idrees Al Asim.
