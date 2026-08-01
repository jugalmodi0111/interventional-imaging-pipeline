# Project Dialygo

An AI Decision-Support Tool for Haemodialysis Vascular Access

*Clinical Orientation & Technical Requirements — a guide for the engineering lead*

**Prepared by:** Dr. G. Gireesh Reddy, Interventional Nephrologist & Vascular Access Specialist

**For:** Sonika Jha, AI Engineering Lead

Version 1.0 · Date: \_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_

**How to read this document.** Part A explains the medical world this tool lives in — no medical background is assumed, and everything is in plain language with analogies. Part B is the technical brief. Please read Part A first; the requirements in Part B only make sense once the clinical picture is clear.

## Part A — Understanding the Clinical World

### A1. The patient: kidney failure and dialysis

Our patients are people whose kidneys have permanently failed — a condition called end-stage kidney disease.

Healthy kidneys are the body's filtration plant: every day they clean the blood, removing waste products and excess water, and keep the body's chemistry balanced. When the kidneys fail, this waste and water build up in the blood. Untreated, this is fatal within weeks.

**Dialysis** is a machine that does the kidneys' job artificially. The patient's blood is drawn out of the body, pumped through a filter that removes the waste and excess water, and returned — clean — to the body. Most patients do this three times a week, about four hours each session, for the rest of their lives (or until a transplant).

> ***Analogy** Think of dialysis as sending the patient's blood through an external washing machine, three times a week, forever. To wash blood fast enough, the machine must pull a large volume out and push it back — far more flow than an ordinary vein can supply.*

### A2. The lifeline: what “vascular access” is

To connect a patient to the dialysis machine, you need a reliable, high-flow doorway into their bloodstream. This doorway is called the vascular access. It is, quite literally, the patient's lifeline: no working access means no dialysis, and no dialysis means death.

An ordinary vein cannot supply enough flow — it would collapse. So we create a special high-flow channel surgically.

> ***Analogy** An ordinary vein is a garden-hose trickle. Dialysis needs a fire-hose flow. So a surgeon plumbs a shortcut that turns the trickle into a torrent.*

### A3. The main type of access: the fistula (AVF)

The best and most common access is an Arteriovenous Fistula (AVF). A surgeon joins an artery directly to a nearby vein, usually in the arm.

Arteries carry blood at high pressure; veins are low-pressure. When they are joined, the high-pressure arterial blood floods into the vein. Over several weeks the vein responds by growing thick, strong, and enlarged — it “matures” — becoming a durable, high-flow channel that can be needled again and again for years.

**The surgical join has a name we will use throughout: the** anastomosis. It is the single most important landmark in every image.

There are two other access types you will occasionally hear about: an AV graft (a synthetic tube used when the patient's own veins are too poor) and a central venous catheter (a tube placed into a large chest vein — a last resort, because it is prone to infection and clotting). Our project focuses on the fistula.

### A4. Why the access matters so much

- **It is the lifeline.** If the access fails, dialysis stops. There is no time to spare.

- **It fails often.** Access problems are the single largest cause of hospitalisation for dialysis patients. Keeping the access working is a constant clinical battle.

- **It is hard to manage.** Deciding whether an access is failing, why, and what to do requires specialist skill — and that skill is scarce (see A8).

So the access is both the most vital and the most fragile part of a dialysis patient's care. That is why a tool that helps manage it can matter enormously.

### A5. The access circuit: a loop from heart to heart

It helps to see the whole access not as a single spot but as a complete plumbing loop. Blood is pumped out by the heart, travels through the access, and returns to the heart. If flow is choked anywhere along this loop, the whole access suffers.

The loop runs from the left ventricle (LV) — the heart's main pump — out through the arteries, across the fistula, back through the veins and the large central veins in the chest, and finally into the right atrium (RA), where blood re-enters the heart.

> ***Analogy** Picture a closed water loop: pump → pipe out → the special junction → pipe back → into the tank. A kink anywhere — near the junction, along the return pipe, or at the tank inlet — slows the whole loop. Our job is to find the kink and know where it is.*

The loop is divided into named segments. This table is your map — you will see these names on every image and in the requirements:

| **Segment (in loop order)** | **What it is, in plain terms** | **What commonly goes wrong there** |
|----|----|----|
| Feeding artery (inflow) | the artery bringing blood toward the fistula | rarely the main problem |
| Anastomosis | the surgical join of artery to vein | narrowing right at the join |
| Juxta-anastomotic | the vein just past the join | the \#1 trouble spot — most narrowings are here |
| Body of the fistula | the long vein segment that gets needled | ballooning (aneurysm), needling damage |
| Cephalic arch / swing point | where the vein bends to join deeper veins | stubborn, tight narrowings |
| Central veins (chest) | large veins returning blood in the chest | narrowing, often after old catheters |
| SVC → right atrium | the final large vein into the heart | narrowing — serious; causes arm/face swelling |

### A6. What goes wrong

**Stenosis (narrowing).** The most common problem. The vessel narrows — like a kink or scale build-up in a pipe — reducing flow and raising pressure. A “significant” stenosis is one narrow enough to matter clinically. This is the main thing our first model will learn to spot.

**Thrombosis (clot).** A narrowing can progress until a clot blocks the vessel completely and the access “dies.” This is an emergency.

**Aneurysm (ballooning).** Repeated needling can weaken the wall so it balloons out.

Why does this happen? A vein was never built to carry high-pressure arterial flow. The turbulence, plus repeated needling, causes scar tissue to form and the vessel to narrow. It is a slow, recurring process — which is why these patients are imaged again and again over the years.

### A7. How we look inside: fistulography (the images we have)

To find a narrowing, we take X-ray movies of the access while injecting a contrast dye that makes the blood vessels visible.

The procedure is done in an endovascular suite / cath lab. A thin tube is placed into the access, dye is injected, and an X-ray camera films the dye flowing through the vessels in real time. Where the vessel is narrowed, the bright column of dye pinches in — that pinch is the stenosis. This imaging is called fistulography, and each recording is a short movie (a “cine loop”) made of many still frames.

**This is our raw material.** Our institute has performed over 2,100 such procedures, and every one is stored. That archive — real, expert-performed vascular-access studies — is the asset this whole project is built on. It is rare and hard to replicate.

Today, a specialist reads these movies and decides three things: is there a significant narrowing? where along the circuit is it? and does it need treatment (opening it with a balloon, called angioplasty)?

### A8. The problem we are solving

Reading a fistulogram well requires an interventional nephrologist — a rare specialist. India has roughly 50 of them for an enormous and growing dialysis population.

Most general nephrologists, who see the majority of dialysis patients, cannot confidently read these images. As a result, patients are referred to the specialist too late, or unnecessarily, and the scarce specialists' time is spent inefficiently.

**Our goal.** Build an AI tool that helps a general nephrologist read a vascular-access image and answer one practical question: does this patient need the specialist? It is a triage aid — it extends the reach of the 50 specialists to patients they could never personally see. It supports the clinician's judgment; it does not replace it, and it does not perform the treatment.

## Part B — What We Are Building (Requirements)

### B1. Purpose and intended use

**The tool is a decision-support / triage aid, not an autonomous diagnostic device.** Intended user: a general nephrologist without interventional training. Purpose: help decide whether a patient needs specialist referral — not to prescribe treatment. The tool's output always supports a clinician's judgment; it never replaces it.

### B2. Scope

**Dataset scope — the full access circuit**

The image library spans the entire access circuit, left ventricle to right atrium (all segments in the Part A map), each labelled normal vs. abnormal. This is the complete clinical vision and the eventual scope of the tool.

**Model scope — staged, one segment at a time**

Trained models are delivered per segment, starting where image data is densest, then combined into a whole-circuit reader. This is arithmetic, not timidity: a segment with only a handful of images cannot be trained or validated. Ambition in the scope, discipline in the build.

**Model One** analyses the juxta-anastomotic segment only (normal vs. significant stenosis), proving the whole pipeline end-to-end before the same recipe rolls across the circuit.

> *Build order, set by image density in a standard fistulogram (fistulograms show the venous/central side densely and the arterial inflow rarely):*

| **Segment (LV → RA)**       | **Image density** | **Build order** |
|-----------------------------|-------------------|-----------------|
| Arterial inflow             | Sparse            | Later           |
| Anastomosis                 | Dense             | Early           |
| Juxta-anastomotic           | Dense             | **FIRST**       |
| Body of AVF / outflow       | Dense             | Early           |
| Cephalic arch / swing point | Medium            | Early–mid       |
| Central AVF veins           | Medium            | Mid             |
| Thoracic central veins      | Medium–thin       | Mid–late        |
| SVC → right atrium          | Thin              | Last            |

### B3. Inputs and outputs

- **Input:** a single still angiographic frame (PNG), de-identified, cropped to the segment of interest.

- **Validity gate:** reject any input that is not a valid vascular-access angiogram (wrong modality, corrupt or unrelated image) rather than attempting to read it.

- **Output:** (a) a triage suggestion — refer / reasonable to observe / uncertain; (b) a calibrated confidence, not a bare yes/no; (c) when confidence is low, default to uncertain / refer for review, never to a false “normal.”

### B4. Technical approach

A frozen foundation-model backbone (DINOv3, or DINOv2 during development) for perception, with a lightweight trained classification head. Alternatives may be proposed, but the sample-efficiency requirement is fixed: the labelled dataset is expert-generated and finite, so the method must learn well from limited data.

### B5. Data handling — mandatory conditions

**This is a non-negotiable section, not a preference.**

- All source images originate from the Institute of Nephro-Urology and are governed by its data-use and ethics approvals. No patient data — identifiable or de-identified — may be copied, transmitted, stored, or processed outside the environment and terms set by the institutional agreement.

- Until that agreement is in place, development uses public, synthetic, or the engineer's own non-patient sample data only.

- Data is split by patient, never by image — no patient may appear in both training and testing.

- The dataset and all models derived from it remain the property of Dr. Reddy / the Institute, per the governing agreement.

### B6. Validation requirements

- Report performance with patient-level splits only.

- External validation is a first-class deliverable: the tool must be tested on images from at least one site other than the source institute before any performance claim or deployment.

- Report where the model fails, not only where it succeeds.

### B7. Ground truth

Labels are provided by the clinical lead, and where feasible by multiple readers with disagreements resolved by consensus, anchored to clinical correlates where available.

**The engineer does not define what counts as “abnormal.” That is a clinical judgment reserved to the clinical lead — the clinical soul of the tool stays with the clinician.**

### B8. Safety and deployment

- Deployment is hosted / central: the tool serves predictions; model weights are not distributed — consistent with the backbone's licence.

- The interface must resist automation bias: it presents a suggestion to weigh, not a conclusion — “assistant, not oracle.”

- The tool is positioned as decision-support Software as a Medical Device (SaMD); its intended-use claim stays within that boundary.

### B9. Deliverables and ownership

- Deliverables: the trained Model One, the source code, the validation report, and documentation. \[Complete this list.\]

**Intellectual property, ownership, and the nature of the engineer's involvement (advisor / contractor / collaborator / co-founder) are defined in a separate written agreement executed before development begins.** This document is technical scope only and does not, by itself, grant any rights to the data, the models, or the Dialygo project.

## Glossary — quick reference

| **Term** | **Plain meaning** |
|----|----|
| Dialysis | a machine that cleans the blood when the kidneys have failed |
| Vascular access | the high-flow doorway into the bloodstream used for dialysis — the patient's lifeline |
| Fistula (AVF) | an artery surgically joined to a vein to create that high-flow channel |
| Anastomosis | the surgical join point of artery and vein — the key landmark |
| Juxta-anastomotic | the vein segment just past the join — the most common site of trouble |
| Stenosis | a narrowing of the vessel — the main problem we detect |
| Thrombosis | a clot fully blocking the vessel — an emergency |
| Fistulography | the X-ray dye study that films the access — our source images |
| Angioplasty | opening a narrowing with a balloon — the specialist's treatment |
| Triage | deciding who needs the specialist — what our tool assists with |
