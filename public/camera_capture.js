(function () {
  "use strict";

  const BUTTON_ID = "smartshop-camera-button";
  const OVERLAY_ID = "smartshop-camera-overlay";

  let buttonMounted = false;
  let activeStream = null;
  let facingMode = "environment";

  const CAMERA_ICON =
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M14.5 4h-5L7 7H4a2 2 0 0 0-2 2v9a2 2 0 0 0 2 2h16a2 2 0 0 0 2-2V9a2 2 0 0 0-2-2h-3l-2.5-3z"/><circle cx="12" cy="13" r="3"/></svg>';

  function findComposerTextarea() {
    return document.querySelector("textarea");
  }

  function findComposerStartAdornment() {
    const textarea = findComposerTextarea();
    if (!textarea) return null;

    const fieldRoot =
      textarea.closest(".MuiTextField-root") ||
      textarea.closest(".MuiInputBase-root") ||
      textarea.parentElement;

    if (!fieldRoot) return null;

    const startAdornment = fieldRoot.querySelector(".MuiInputAdornment-positionStart");
    if (startAdornment) return startAdornment;

    const chatInput = document.getElementById("chat-input");
    if (chatInput) {
      const composer = chatInput.closest(".rounded-3xl") || chatInput.parentElement?.parentElement;
      if (!composer) return null;
      const toolbars = composer.querySelectorAll(".flex.items-center");
      for (const toolbar of toolbars) {
        if (toolbar.querySelector("button")) {
          return toolbar;
        }
      }
    }

    return null;
  }

  function findComposerFileInput() {
    const textarea = findComposerTextarea();
    const scope = textarea?.closest(".MuiTextField-root") || document;
    const inputs = scope.querySelectorAll('input[type="file"]');
    if (inputs.length > 0) {
      return inputs[inputs.length - 1];
    }
    return document.querySelector('input[type="file"]');
  }

  function findSubmitButton() {
    const textarea = findComposerTextarea();
    if (!textarea) return null;

    const fieldRoot = textarea.closest(".MuiTextField-root");
    if (fieldRoot) {
      const endAdornment = fieldRoot.querySelector(".MuiInputAdornment-positionEnd");
      const endButton = endAdornment?.querySelector("button");
      if (endButton) return endButton;
    }

    const composer = textarea.closest(".rounded-3xl");
    if (composer) {
      const submitCandidates = composer.querySelectorAll("button");
      if (submitCandidates.length > 0) {
        return submitCandidates[submitCandidates.length - 1];
      }
    }

    return null;
  }

  function setTextareaValue(text) {
    const textarea = findComposerTextarea();
    if (!textarea) return false;

    const setter = Object.getOwnPropertyDescriptor(
      window.HTMLTextAreaElement.prototype,
      "value"
    )?.set;
    if (!setter) return false;

    setter.call(textarea, text);
    textarea.dispatchEvent(new Event("input", { bubbles: true }));
    textarea.dispatchEvent(new Event("change", { bubbles: true }));
    return true;
  }

  function dataUrlToFile(dataUrl, filename) {
    const parts = dataUrl.split(",");
    const mimeMatch = parts[0].match(/:(.*?);/);
    const mime = mimeMatch ? mimeMatch[1] : "image/jpeg";
    const binary = atob(parts[1]);
    const bytes = new Uint8Array(binary.length);
    for (let i = 0; i < binary.length; i += 1) {
      bytes[i] = binary.charCodeAt(i);
    }
    return new File([bytes], filename, { type: mime });
  }

  function uploadCapturedFile(file) {
    const fileInput = findComposerFileInput();
    if (!fileInput) {
      throw new Error("Could not find the chat upload input.");
    }

    const transfer = new DataTransfer();
    transfer.items.add(file);
    fileInput.files = transfer.files;
    fileInput.dispatchEvent(new Event("change", { bubbles: true }));
  }

  function waitForSubmitReady(timeoutMs) {
    const deadline = Date.now() + (timeoutMs || 20000);

    return new Promise(function (resolve, reject) {
      function check() {
        const submitButton = findSubmitButton();
        const textarea = findComposerTextarea();
        const textareaDisabled = textarea?.disabled;

        if (submitButton && !submitButton.disabled && !textareaDisabled) {
          resolve();
          return;
        }

        if (Date.now() > deadline) {
          reject(new Error("Image upload timed out. Please try again."));
          return;
        }

        window.setTimeout(check, 200);
      }

      check();
    });
  }

  async function submitCapturedPhoto(dataUrl) {
    const file = dataUrlToFile(dataUrl, "camera_capture_" + Date.now() + ".jpg");
    uploadCapturedFile(file);
    await waitForSubmitReady();
    setTextareaValue(" ");

    await new Promise(function (resolve) {
      window.setTimeout(resolve, 100);
    });

    const submitButton = findSubmitButton();
    if (!submitButton) {
      throw new Error("Could not find the send button.");
    }
    submitButton.click();
  }

  function mountCameraButton() {
    if (buttonMounted) return;

    const adornment = findComposerStartAdornment();
    if (!adornment) return;

    if (document.getElementById(BUTTON_ID)) {
      buttonMounted = true;
      return;
    }

    const button = document.createElement("button");
    button.id = BUTTON_ID;
    button.type = "button";
    button.title = "Capture photo from camera";
    button.setAttribute("aria-label", "Capture photo from camera");
    button.innerHTML = CAMERA_ICON;
    button.addEventListener("click", openCameraModal);

    const iconButtons = adornment.querySelectorAll("button");
    if (iconButtons.length > 0) {
      const insertBefore = iconButtons[1] || null;
      adornment.insertBefore(button, insertBefore);
    } else {
      adornment.appendChild(button);
    }

    buttonMounted = true;
  }

  function stopStream() {
    if (!activeStream) return;
    activeStream.getTracks().forEach(function (track) {
      track.stop();
    });
    activeStream = null;
  }

  function setStatus(message) {
    const status = document.getElementById("smartshop-camera-status");
    if (status) status.textContent = message || "";
  }

  function closeCameraModal() {
    stopStream();
    const overlay = document.getElementById(OVERLAY_ID);
    if (overlay) overlay.remove();
  }

  async function startPreview(video) {
    stopStream();
    setStatus("Requesting camera access...");

    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      throw new Error("Camera access is not supported in this browser.");
    }

    activeStream = await navigator.mediaDevices.getUserMedia({
      audio: false,
      video: {
        facingMode: facingMode,
        width: { ideal: 1280 },
        height: { ideal: 960 },
      },
    });

    video.srcObject = activeStream;
    await video.play();
    setStatus("Align your product in the frame, then tap Capture.");
  }

  function buildModal() {
    const overlay = document.createElement("div");
    overlay.id = OVERLAY_ID;
    overlay.setAttribute("role", "dialog");
    overlay.setAttribute("aria-modal", "true");
    overlay.setAttribute("aria-label", "Camera capture");

    overlay.innerHTML =
      '<div id="smartshop-camera-dialog">' +
      '  <div id="smartshop-camera-header">' +
      "    <span>Take a product photo</span>" +
      '    <button id="smartshop-camera-close" type="button" aria-label="Close camera">×</button>' +
      "  </div>" +
      '  <div id="smartshop-camera-preview-wrap">' +
      '    <video id="smartshop-camera-preview" autoplay playsinline muted></video>' +
      '    <img id="smartshop-camera-still" alt="Captured preview" />' +
      "  </div>" +
      '  <div id="smartshop-camera-status"></div>' +
      '  <div id="smartshop-camera-actions">' +
      '    <button class="smartshop-camera-action" id="smartshop-camera-switch" type="button">Switch camera</button>' +
      '    <button class="smartshop-camera-action" id="smartshop-camera-retake" type="button" hidden>Retake</button>' +
      '    <button class="smartshop-camera-action primary" id="smartshop-camera-capture" type="button">Capture</button>' +
      '    <button class="smartshop-camera-action primary" id="smartshop-camera-send" type="button" hidden>Search with photo</button>' +
      "  </div>" +
      "</div>";

    overlay.addEventListener("click", function (event) {
      if (event.target === overlay) closeCameraModal();
    });

    return overlay;
  }

  async function openCameraModal() {
    if (document.getElementById(OVERLAY_ID)) return;

    const overlay = buildModal();
    document.body.appendChild(overlay);

    const video = document.getElementById("smartshop-camera-preview");
    const still = document.getElementById("smartshop-camera-still");
    const captureButton = document.getElementById("smartshop-camera-capture");
    const sendButton = document.getElementById("smartshop-camera-send");
    const retakeButton = document.getElementById("smartshop-camera-retake");
    const switchButton = document.getElementById("smartshop-camera-switch");
    const closeButton = document.getElementById("smartshop-camera-close");

    let capturedDataUrl = null;

    closeButton.addEventListener("click", closeCameraModal);

    switchButton.addEventListener("click", async function () {
      facingMode = facingMode === "environment" ? "user" : "environment";
      capturedDataUrl = null;
      still.style.display = "none";
      video.style.display = "block";
      captureButton.hidden = false;
      sendButton.hidden = true;
      retakeButton.hidden = true;
      try {
        await startPreview(video);
      } catch (error) {
        setStatus(error.message || "Unable to switch camera.");
      }
    });

    retakeButton.addEventListener("click", async function () {
      capturedDataUrl = null;
      still.style.display = "none";
      video.style.display = "block";
      captureButton.hidden = false;
      sendButton.hidden = true;
      retakeButton.hidden = true;
      try {
        await startPreview(video);
      } catch (error) {
        setStatus(error.message || "Unable to restart camera.");
      }
    });

    captureButton.addEventListener("click", function () {
      if (!video.videoWidth || !video.videoHeight) {
        setStatus("Camera is still starting. Please wait a moment.");
        return;
      }

      const canvas = document.createElement("canvas");
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      const context = canvas.getContext("2d");
      if (!context) {
        setStatus("Could not capture image from camera.");
        return;
      }

      context.drawImage(video, 0, 0, canvas.width, canvas.height);
      capturedDataUrl = canvas.toDataURL("image/jpeg", 0.9);
      still.src = capturedDataUrl;
      still.style.display = "block";
      video.style.display = "none";
      stopStream();

      captureButton.hidden = true;
      sendButton.hidden = false;
      retakeButton.hidden = false;
      setStatus("Preview ready. Send this photo to search your catalog.");
    });

    sendButton.addEventListener("click", async function () {
      if (!capturedDataUrl) {
        setStatus("Capture a photo before sending.");
        return;
      }
      
      const cameraButton = document.getElementById(BUTTON_ID);
      if (cameraButton) cameraButton.disabled = true;
      sendButton.disabled = true;
      setStatus("Uploading photo and starting search...");
      
      try {
        closeCameraModal();
        await submitCapturedPhoto(capturedDataUrl);
      } catch (error) {
        setStatus(error.message || "Could not send the captured photo.");
        sendButton.disabled = false;
      } finally {
        window.setTimeout(function () {
          if (cameraButton) cameraButton.disabled = false;
        }, 1200);
      }
    });

    try {
      await startPreview(video);
    } catch (error) {
      setStatus(error.message || "Unable to access the camera.");
      captureButton.disabled = true;
      switchButton.disabled = true;
    }
  }

  function init() {
    mountCameraButton();
  }

  const observer = new MutationObserver(init);
  observer.observe(document.documentElement, {
    childList: true,
    subtree: true,
  });

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
