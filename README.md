# Capstone-Project

# CareLoop Clinics — No-Show Guard

Repository URL: <YOUR-GITHUB-REPO-URL>

## Summary Findings
- Part 1 (Data Foundations): Cleaning the 1,020 raw booking rows (dropping 20 exact duplicates, 6 unreadable dates, and 4 blank statuses; imputing 17 missing/invalid ages; capping 26 distance and 3 fee outliers) yielded 990 valid appointments with a 22.0% no-show rate (218 no-shows).
- Part 2 (SQL Analytics): Bookings booked 31+ days ahead have a 60.5% no-show rate (26 of 43) versus 16.2% for 1–7 days ahead, while among clinics with at least 95 bookings, Hadapsar Care Centre leads at 27.3% (39 of 143).
- Part 3 (Machine Learning): A standardized Logistic Regression pipeline achieved 0.7737 5-fold cross-validated training ROC-AUC (0.7889 test ROC-AUC) and, at a 0.35 probability threshold, flagged 49 test appointments with 0.5714 precision and 0.5714 recall.
- Part 4 (GenAI Follow-Up Notes): Across the first 10 call-centre notes (9 non-blank API calls plus 1 skipped blank note `N004`), structured extraction with Pydantic validation captured forgotten/no-reminder cases (such as `N001` and `N007`, both under `reminder_sent = No`), transport/distance barriers, work schedule conflicts, and fee concerns.

## Recommendation
CareLoop Clinics should deploy the Logistic Regression risk scorer on the evening before appointments at a 0.35 threshold (flagging ~49 of 198 test bookings, achieving 0.5714 recall and 0.5714 precision at 0.7889 test ROC-AUC) so front-desk staff prioritise high-risk bookings—especially those with `lead_days` in the 15–30 day (30.0% no-show rate) or 31+ day (60.5% no-show rate) bands and those missing reminders (`reminder_sent = No`, where a Yes reminder is associated with 0.3094 odds, or ~69.1% lower odds, holding other model features constant). In parallel, CareLoop should target high-no-show clinics such as Hadapsar Care Centre (27.3%) and Malviya Nagar Clinic (26.8%), confirm reminders ahead of long-lead and high-fee/far-distance visits, and offer flexible evening/Saturday or lower-cost follow-up options where schedule or fee friction arises.

## Chosen Model & Threshold
We selected the `StandardScaler` + `LogisticRegression(max_iter=1000)` pipeline because it achieved the highest 5-fold cross-validated ROC-AUC on the training set (**0.7737**), outperforming the tuned Random Forest (`0.7673`), default Random Forest (`0.7466`), and depth-3 Decision Tree (`0.6784`), while avoiding the severe overfitting of an unpruned tree (`1.0000` train accuracy vs. `0.6717` test accuracy). We chose threshold **0.35** over `0.50` using out-of-fold training predictions because it improved F1 (`0.4892` vs. `0.3471`) and raised recall (`0.4675` vs. `0.2485`), capturing more true no-shows at the modest cost of extra evening confirmation calls.

## Limitations
1. **Small, illustrative call-centre note sample:** Analysing only the first 10 notes (`N001`–`N010`) illustrates structured extraction rather than establishing clinic-wide root-cause shares across all 218 no-shows.
2. **Pre-split cleaning statistics and random split:** Imputation medians, IQR outlier fences, and a random 80/20 train/test split (rather than a strict past-to-future split or patient-grouped split) can introduce minor data leakage across bookings or repeat patients.
3. **Observational data and capacity-dependent threshold:** Associations in historical booking logs do not prove causation (for example, `reminder_sent` is observational), and in production the evening call threshold must be tuned to front-desk calling capacity versus empty-slot costs.

