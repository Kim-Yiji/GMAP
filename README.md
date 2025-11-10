✅ CODING AGENT SYSTEM PROMPT

Context / Background

We are working on a project using the SDD (Stanford Drone Dataset), focusing on trajectory visualization based on threat scores.
We already have a DMRGCN model implementation in the directory yiji/, and the dataset (videos + annotations) in sdd_bookstore/.
Originally, DMRGCN uses 2 relational graphs, but the research direction requires extending this to 4 relational graphs to represent different interaction types.

However, the new model training is not completed yet, so for now we will compute threat scores heuristically, not with the new learned weights.

⸻

🎯 Goal

For a given target pedestrian, compute the threat score from every other obstacle (pedestrians, bicycles, etc.) in each frame and overlay that score as text on the original SDD video — one score per interaction.

We also need to store the score and text position per frame so that later visualization frameworks (possibly OpenCV / matplotlib / Blender / Unity) can use the data.

⸻

📥 Inputs / Directory Structure

/sdd_bookstore/
    videos/
    annotations/

/yiji/
    dmr_gcn_model_files/
    checkpoints/  # trained model exists here


⸻

📌 Threat Score Calculation (Temporary / Manual)

We will compute threat scores using a dummy uniform weighting:

threat_score = sigmoid( w1*f1 + w2*f2 + w3*f3 + w4*f4 )

Where:
	•	w1 = w2 = w3 = w4 = 1
	•	f1, f2, f3, f4 correspond to the 4 relational-effect features (can be distance, velocity difference, heading alignment, object class interaction, etc. — use simple placeholder features for now).
	•	Use sigmoid to keep output in [0,1].

⸻

🎥 Visualization Requirement

For every frame:
	•	Identify target pedestrian coordinates.
	•	For every other tracked object, compute threat score.
	•	Draw text overlay (ID, score) near that object in the frame.
	•	Save frame-by-frame metadata:

{
   "frame_id": ...,
   "target_id": ...,
   "interactions": [
       { "object_id": ..., "score": ..., "position": (x,y) },
       ...
   ]
}

	•	Store logs in a usable format: .json or .csv.

⸻

📦 Required Output
	1.	A Python script/module that:
	•	Loads a video + corresponding annotation.
	•	Extracts positions of target and other objects per frame.
	•	Computes threat scores using temporary uniform-weight method.
	•	Overlays the text onto the video frames (OpenCV).
	•	Saves:
	•	Annotated video output (.mp4).
	•	Metadata file storing threat scores + screen positions.
	2.	Clear inline comments and function structure.

⸻

🧭 Step-By-Step Plan for Implementation
	1.	Load annotations (likely .txt or .csv) and parse object positions per frame.
	2.	Select one pedestrian as the target pedestrian (can hardcode ID or allow selection).
	3.	For each frame:
	•	Gather all present object bounding box centers.
	•	Compute relational features (distance, relative speed, orientation difference, class interaction).
	•	Apply weighted sum + sigmoid.
	4.	Visualize threat score using OpenCV cv2.putText.
	5.	Save log record for each interaction and frame into JSON.
	6.	Output processed video and metadata.

⸻

🧑‍💻 Coding Requirements
	•	Use Python 3.8+
	•	Use OpenCV for video rendering
	•	Use numpy for calculations
	•	Ensure code is modular (not a single giant script)
	•	Include CLI usage example
	•	Do not rely on new model training — use manual scoring only

⸻

✅ Final Task

Generate the complete implementation described above, including:
	•	Code structure
	•	Data parsing
	•	Frame loop
	•	Threat score computation
	•	Visualization
	•	Metadata saving

If needed, ask clarification questions before coding, rather than guessing.