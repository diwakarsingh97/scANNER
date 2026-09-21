import {
    FaceLandmarker,
    FilesetResolver
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.22/vision_bundle.mjs";


/* =========================================
   ELEMENTS
========================================= */

const video = document.getElementById("video");
const canvas = document.getElementById("canvas");

const startButton =
    document.getElementById("startButton");

const systemStatus =
    document.getElementById("systemStatus");

const cameraMessage =
    document.getElementById("cameraMessage");

const faceStatus =
    document.getElementById("faceStatus");

const expression =
    document.getElementById("expression");

const confidence =
    document.getElementById("confidence");

const confidenceFill =
    document.getElementById("confidenceFill");

const ctx = canvas.getContext("2d");


/* =========================================
   VARIABLES
========================================= */

let faceLandmarker = null;

let cameraRunning = false;

let lastVideoTime = -1;

let animationFrame = null;


/* =========================================
   MEDIAPIPE MODEL
========================================= */

async function createFaceLandmarker() {

    systemStatus.textContent =
        "LOADING AI MODEL";

    cameraMessage.textContent =
        "LOADING AI VISION...";


    try {

        const vision =
            await FilesetResolver.forVisionTasks(
                "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.22/wasm"
            );


        faceLandmarker =
            await FaceLandmarker.createFromOptions(
                vision,
                {
                    baseOptions: {

                        modelAssetPath:
                            "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",

                        delegate: "GPU"

                    },

                    runningMode: "VIDEO",

                    numFaces: 1,

                    minFaceDetectionConfidence: 0.5,

                    minFacePresenceConfidence: 0.5,

                    minTrackingConfidence: 0.5,

                    outputFaceBlendshapes: true
                }
            );


        systemStatus.textContent =
            "AI MODEL READY";

        cameraMessage.textContent =
            "PRESS START SCANNER";

        startButton.disabled = false;

    }

    catch (error) {

        console.error(error);

        systemStatus.textContent =
            "MODEL ERROR";

        cameraMessage.textContent =
            "AI MODEL FAILED TO LOAD";

        faceStatus.textContent =
            "ERROR";

        startButton.disabled = true;

    }
}


/* =========================================
   CAMERA
========================================= */

async function startCamera() {

    if (!faceLandmarker) {

        alert(
            "AI model is still loading. Please wait a moment."
        );

        return;
    }


    try {

        const stream =
            await navigator.mediaDevices.getUserMedia({

                video: {

                    facingMode: "user",

                    width: {
                        ideal: 1280
                    },

                    height: {
                        ideal: 720
                    }

                },

                audio: false

            });


        video.srcObject = stream;


        await video.play();


        cameraRunning = true;


        startButton.textContent =
            "SCANNER ACTIVE";

        startButton.disabled = true;


        systemStatus.textContent =
            "SCANNING";

        cameraMessage.style.display =
            "none";


        resizeCanvas();


        detectFace();


    }

    catch (error) {

        console.error(error);

        systemStatus.textContent =
            "CAMERA ERROR";

        cameraMessage.style.display =
            "flex";

        cameraMessage.textContent =
            "CAMERA ACCESS DENIED";

        faceStatus.textContent =
            "NO CAMERA";

        alert(
            "Camera could not be started. Please allow camera permission and reload the page."
        );

    }

}


/* =========================================
   RESIZE CANVAS
========================================= */

function resizeCanvas() {

    if (!video.videoWidth) {
        return;
    }


    canvas.width =
        video.videoWidth;

    canvas.height =
        video.videoHeight;

}


/* =========================================
   FACE DETECTION LOOP
========================================= */

function detectFace() {

    if (!cameraRunning) {
        return;
    }


    if (
        video.readyState >= 2 &&
        video.currentTime !== lastVideoTime
    ) {

        lastVideoTime =
            video.currentTime;


        const now =
            performance.now();


        const result =
            faceLandmarker.detectForVideo(
                video,
                now
            );


        processResult(result);

    }


    animationFrame =
        requestAnimationFrame(detectFace);

}


/* =========================================
   PROCESS FACE
========================================= */

function processResult(result) {

    ctx.clearRect(
        0,
        0,
        canvas.width,
        canvas.height
    );


    if (
        !result.faceLandmarks ||
        result.faceLandmarks.length === 0
    ) {

        faceStatus.textContent =
            "SEARCHING";

        expression.textContent =
            "—";

        confidence.textContent =
            "0%";

        confidenceFill.style.width =
            "0%";

        return;
    }


    const landmarks =
        result.faceLandmarks[0];


    faceStatus.textContent =
        "HUMAN DETECTED";


    /* -------------------------------------
       DRAW LANDMARKS
    ------------------------------------- */

    drawLandmarks(landmarks);


    /* -------------------------------------
       FACE BOX
    ------------------------------------- */

    drawFaceBox(landmarks);


    /* -------------------------------------
       CONFIDENCE
    ------------------------------------- */

    let conf = 0.85;


    if (
        result.faceBlendshapes &&
        result.faceBlendshapes.length > 0
    ) {

        const categories =
            result.faceBlendshapes[0].categories;


        const values =
            categories.map(
                item => item.score
            );


        if (values.length > 0) {

            conf =
                Math.min(
                    0.99,
                    Math.max(
                        0.75,
                        values.reduce(
                            (a, b) => a + b,
                            0
                        ) / values.length + 0.75
                    )
                );

        }


        const detectedExpression =
            detectExpression(categories);


        expression.textContent =
            detectedExpression;

    }


    const percentage =
        Math.round(conf * 100);


    confidence.textContent =
        `${percentage}%`;

    confidenceFill.style.width =
        `${percentage}%`;

}


/* =========================================
   DRAW LANDMARKS
========================================= */

function drawLandmarks(landmarks) {

    ctx.save();


    ctx.fillStyle =
        "#00f6ff";


    for (const point of landmarks) {

        const x =
            point.x * canvas.width;

        const y =
            point.y * canvas.height;


        ctx.beginPath();

        ctx.arc(
            x,
            y,
            1.5,
            0,
            Math.PI * 2
        );

        ctx.fill();

    }


    ctx.restore();

}


/* =========================================
   DRAW FACE BOX
========================================= */

function drawFaceBox(landmarks) {

    let minX = 1;
    let minY = 1;

    let maxX = 0;
    let maxY = 0;


    for (const point of landmarks) {

        minX =
            Math.min(minX, point.x);

        minY =
            Math.min(minY, point.y);

        maxX =
            Math.max(maxX, point.x);

        maxY =
            Math.max(maxY, point.y);

    }


    const padding = 0.03;


    minX =
        Math.max(0, minX - padding);

    minY =
        Math.max(0, minY - padding);

    maxX =
        Math.min(1, maxX + padding);

    maxY =
        Math.min(1, maxY + padding);


    const x =
        minX * canvas.width;

    const y =
        minY * canvas.height;

    const width =
        (maxX - minX) *
        canvas.width;

    const height =
        (maxY - minY) *
        canvas.height;


    ctx.save();


    ctx.strokeStyle =
        "#00f6ff";

    ctx.lineWidth = 2;

    ctx.shadowColor =
        "#00f6ff";

    ctx.shadowBlur = 12;


    ctx.strokeRect(
        x,
        y,
        width,
        height
    );


    ctx.restore();

}


/* =========================================
   EXPRESSION DETECTION
========================================= */

function detectExpression(categories) {

    const scores = {};


    for (const item of categories) {

        scores[item.categoryName] =
            item.score;

    }


    const smile =
        average(
            scores.mouthSmileLeft || 0,
            scores.mouthSmileRight || 0
        );


    const frown =
        average(
            scores.mouthFrownLeft || 0,
            scores.mouthFrownRight || 0
        );


    const jawOpen =
        scores.jawOpen || 0;


    const eyeWide =
        average(
            scores.eyeWideLeft || 0,
            scores.eyeWideRight || 0
        );


    if (
        smile > 0.45
    ) {

        return "HAPPY";

    }


    if (
        jawOpen > 0.45 &&
        eyeWide > 0.25
    ) {

        return "SURPRISED";

    }


    if (
        frown > 0.35
    ) {

        return "SAD";

    }


    return "NEUTRAL";

}


/* =========================================
   AVERAGE
========================================= */

function average(a, b) {

    return (a + b) / 2;

}


/* =========================================
   START BUTTON
========================================= */

startButton.addEventListener(
    "click",
    startCamera
);


/* =========================================
   VIDEO RESIZE
========================================= */

video.addEventListener(
    "loadedmetadata",
    resizeCanvas
);

window.addEventListener(
    "resize",
    resizeCanvas
);


/* =========================================
   INITIALIZE
========================================= */

startButton.disabled = true;

createFaceLandmarker();
