# gemini-agent-assist
Includes starter template to make complex agents with Gemini  


Enabling Vertex AI
```
export GCP_PROJECT_ID="hackathon0-project"
export SERVICE_AC_DISPLAYNAME="hackathon0-vertex-sa"
export LOCATION="us-central1"

gcloud iam service-accounts create $SERVICE_AC_DISPLAYNAME --display-name $SERVICE_AC_DISPLAYNAME

gcloud services enable cloudresourcemanager.googleapis.com \
    artifactregistry.googleapis.com \
    iam.googleapis.com \
    storage.googleapis.com \
    aiplatform.googleapis.com \
    --project=$GCP_PROJECT_ID


for role in resourcemanager.projectIamAdmin \
            iam.serviceAccountUser \
            run.admin \
            artifactregistry.writer \
            artifactregistry.reader \
            artifactregistry.admin \
            storage.admin \
            storage.objectAdmin \
            storage.objectViewer \
            storage.objectCreator \
            aiplatform.user \
            aiplatform.admin; do
    gcloud projects add-iam-policy-binding $GCP_PROJECT_ID \
        --member=serviceAccount:$SERVICE_AC_DISPLAYNAME@$GCP_PROJECT_ID.iam.gserviceaccount.com \
        --role=roles/$role
done



gcloud iam service-accounts keys create key.json \
    --iam-account=$SERVICE_AC_DISPLAYNAME@$GCP_PROJECT_ID.iam.gserviceaccount.com

```

```
import os
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "../key.json"
```