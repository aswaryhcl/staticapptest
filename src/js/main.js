// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license.



//var inputdict = [{skill:null}];
sessionStorage.clear();
var inputdict = {};
var inputarray = [];


const TTSVoice = "en-US-JennyMultilingualNeural" // Update this value if you want to use a different voice

const CogSvcRegion = "eastus" // Fill your Azure cognitive services region here, e.g. westus2

const IceServerUrl = "turn:relay.communication.microsoft.com:3478" // Fill your ICE server URL here, e.g. turn:turn.azure.com:3478
let IceServerUsername
let IceServerCredential

// This is the only avatar which supports live streaming so far, please don't modify
const TalkingAvatarCharacter = "lisa"
const TalkingAvatarStyle = "casual-sitting"

supported_languages = ["en-US", "de-DE", "zh-CN", "ar-AE"] // The language detection engine supports a maximum of 4 languages

const BackgroundColor = '#FFFFFFFF'
//const BackgroundColor = 'transparent'

let token

const speechSynthesisConfig = SpeechSDK.SpeechConfig.fromEndpoint(new URL("wss://{region}.tts.speech.microsoft.com/cognitiveservices/websocket/v1?enableTalkingAvatar=true".replace("{region}", CogSvcRegion)))

// Global objects
var speechSynthesizer
var peerConnection
var previousAnimationFrameTimestamp = 0

let skill = sessionStorage.getItem("skill");

if (skill==='cka') {
  var system_prompt = `You are an AI assistant called CKA aka cognitive Knowledge Assistant, which provides answers from a private knowledgebase or database.
  - Before calling a function, aim to answer queries using existing conversational context.
  - Any question referred to CKA should go through the function referCkaKnowledgeBase
  - If the  information isn't clear or available, consult referCkaKnowledgeBase for accurate details. Never invent answers.  
  - Before seeking specifics, scan previous parts of the conversation. Reuse information if available, avoiding repetitive queries.
  - Only use the functions you have been provided with.
  - NEVER GUESS FUNCTION INPUTS! If a user's request is unclear, request further clarification. 
  - Provide responses within 3 sentences, emphasizing conciseness and accuracy.
  - If not specified otherwise, ALWAYS consult referCkaKnowledgeBase function. 
  - Pay attention to the language the customer is using in their latest statement and respond in the same language!
  - If any technical support is required, consult referCkaKnowledgeBase for accurate details.
  - If no product is mentioned in Technical support, the product is cognitive knowledge assistant (CKA) developed by HCL.
  - Always provide responses in a human way in natural language and not pointers like a robot.
  - When USER_QUESTION is a Greeting or a statement or a generic question not related to CKA , NEVER CONSULT referCkaKnowledgeBase.
  `
}
else if (skill === 'dbquery') {
  var system_prompt = `You are an AI assistant called CKA aka cognitive Knowledge Assistant, which provides answers from a database.
  - Before calling a function, aim to answer queries using existing conversational context.
  - Any question related to Orders or Products, then DO NOT refer the function referCkaKnowledgeBase
  - If the  information isn't clear or available, ask user for accurate details. Never invent answers.  
  - Before seeking specifics, scan previous parts of the conversation. Reuse information if available, avoiding repetitive queries.
  - Only use the functions you have been provided with.
  - NEVER GUESS FUNCTION INPUTS! If a user's request is unclear, request further clarification. 
  - Provide responses within 3 sentences at max, emphasizing conciseness and accuracy.
  - Pay attention to the language the customer is using in their latest statement and respond in the same language!
  - If no account is mentioned, the default ACCOUNT_ID is 1000 
  - Always provide responses in a human way in natural language and not pointers like a robot.
  - When USER_QUESTION is a Greeting or a statement or a generic question , NEVER CONSULT any functios
  `
}
else if (skill==='analyticsinsights') {
  var system_prompt = `You are a Snowflake Data expert, and your job is to either generate a valid SQL query compatible with snowflake if provided with a data model OR explain the results generated by the query execution.
  - Never assume data model and column names or sql queries or data values.
  - From the data model provided, Identify the Tables, columns of each table in tabular format and identify the Primary Key and Foreign Key
  - Use the data model of the schema provided as a bounded context to answer the user's question.
  - Always use only valid column names provided in the data model.
  - If data type of column is varchar, convert the values to lower case.
  - Always explain the query results in simple language as a Data Analyst.
  - If the  information isn't clear or available, Ask the user for accurate details. Never invent answers.  
  - Before seeking specifics, scan previous parts of the conversation. Reuse information if available, avoiding repetitive queries.
  - Only use the functions you have been provided with.
  - NEVER GUESS FUNCTION INPUTS! If a user's request is unclear, request further clarification. 
  - While Providing responses always emphasize on conciseness and accuracy. 
  - Pay attention to the language the customer is using in their latest statement and respond in the same language!
  - Always provide responses in a human way in natural language and not pointers like a robot.
  - When USER_QUESTION is a Greeting or a statement or a generic question , NEVER CONSULT any function.
  `
}
else {
  var system_prompt = `You are an AI assistant who answers users queries in a polite manner.
  - Before calling a function, aim to answer queries using existing conversational context.
  - If the  information isn't clear or available, ask user for accurate details. Never invent answers.  
  - Before seeking specifics, scan previous parts of the conversation. Reuse information if available, avoiding repetitive queries. 
  - Provide responses within 3 sentences at max, emphasizing conciseness and accuracy.
  - Pay attention to the language the customer is using in their latest statement and respond in the same language!
  - Always provide responses in a human way in natural language and not pointers like a robot.
  - When USER_QUESTION is a Greeting or a statement or a generic question , NEVER CONSULT any functios
  `
}

messages = [{ "role": "system", "content": system_prompt }];



function selectSkill() {
  sessionStorage.clear();
  var skill = document.getElementById("skill").value;
  sessionStorage.setItem("skill",skill);
  let skillSelected = sessionStorage.getItem("skill");
  console.log('skill Selected: ',skillSelected)
}


function removeDocumentReferences(str) {
  // Regular expression to match [docX]
  var regex = /\[doc\d+\]/g;

  // Replace document references with an empty string
  var result = str.replace(regex, '');

  return result;
}

// Setup WebRTC
function setupWebRTC() {
  // Create WebRTC peer connection
  //fetch("/api/getIceServerToken", {
    //method: "POST"
  //})
    //.then(response => response.json())
    //.then(response => { 
      IceServerUsername = "BQAANmXAyIAB2iE0CgIjuChTUuN6ju7NH2owrtXiS1AAAAAMARBLzcgb+8ZGv7VTu51ROGIsrn3j1xkOsVZBYYwYaz6M5IQwJe4="
      IceServerCredential = "33qDidv0KCP3VDTvpWZCeSaDq2Y="

      peerConnection = new RTCPeerConnection({
        iceServers: [{
          urls: [IceServerUrl],
          username: IceServerUsername,
          credential: IceServerCredential
        }]
      })
    
      // Fetch WebRTC video stream and mount it to an HTML video element
      peerConnection.ontrack = function (event) {
        console.log('peerconnection.ontrack', event)
        // Clean up existing video element if there is any
        remoteVideoDiv = document.getElementById('remoteVideo')
        for (var i = 0; i < remoteVideoDiv.childNodes.length; i++) {
          if (remoteVideoDiv.childNodes[i].localName === event.track.kind) {
            remoteVideoDiv.removeChild(remoteVideoDiv.childNodes[i])
          }
        }
    
        const videoElement = document.createElement(event.track.kind)
        videoElement.id = event.track.kind
        videoElement.srcObject = event.streams[0]
        videoElement.autoplay = true
        videoElement.controls = false
        document.getElementById('remoteVideo').appendChild(videoElement)

        canvas = document.getElementById('canvas')
        remoteVideoDiv.hidden = true
        canvas.hidden = false

        videoElement.addEventListener('play', () => {
          remoteVideoDiv.style.width = videoElement.videoWidth / 2 + 'px'
          window.requestAnimationFrame(makeBackgroundTransparent)
      })
      }
    
      // Make necessary update to the web page when the connection state changes
      peerConnection.oniceconnectionstatechange = e => {
        console.log("WebRTC status: " + peerConnection.iceConnectionState)
    
        if (peerConnection.iceConnectionState === 'connected') {
          greeting()
          document.getElementById('loginOverlay').classList.add("hidden");
        }
    
        if (peerConnection.iceConnectionState === 'disconnected') {
        }
      }
    
      // Offer to receive 1 audio, and 1 video track
      peerConnection.addTransceiver('video', { direction: 'sendrecv' })
      peerConnection.addTransceiver('audio', { direction: 'sendrecv' })
    
      // Set local description
      peerConnection.createOffer().then(sdp => {
        peerConnection.setLocalDescription(sdp).then(() => { setTimeout(() => { connectToAvatarService() }, 1000) })
      }).catch(console.log)
 
}

async function generateText(prompt) {

  messages.push({
    role: 'user',
    content: prompt
  });

  let skill = sessionStorage.getItem("skill");
  console.log('skill:', skill);
  console.log('messages:', messages);
  inputdict.skill=skill;
  inputdict.messages=messages;
  console.log('inputdict:', inputdict);
  inputarray.push(inputdict);
  console.log('inputarray:', inputarray);

  let generatedText
  let products
  console.log(`Input Message: ${JSON.stringify(inputdict)}`);


  await fetch(`/api/message`, { method: 'POST', headers: { 'Content-Type': 'application/json'}, body: JSON.stringify(inputdict) })
    .then(response => response.json())
    .then(data => {
      generatedText = data["messages"][data["messages"].length - 1].content;
      messages = data["messages"];
      products = data["products"]
    })
    .catch(error => {
        throw(error);
    });
    


    //.then(console.log);

  addToConversationHistory(generatedText, 'light');
  if(products.length > 0) {
    console.log("Product data: " + products[0]);
    addProductToChatHistory(products[0]);
  }
  return generatedText;
}

// Connect to TTS Avatar API
function connectToAvatarService() {
  // Construct TTS Avatar service request
  let videoCropTopLeftX = 600
  let videoCropBottomRightX = 1320
  let backgroundColor = '#00FF00FF'
  // letbackgroundColor = 'transparent'

  console.log(peerConnection.localDescription)
  const clientRequest = {
    protocol: {
      name: "WebRTC",
      webrtcConfig: {
        clientDescription: btoa(JSON.stringify(peerConnection.localDescription)),
        iceServers: [{
          urls: [IceServerUrl],
          username: IceServerUsername,
          credential: IceServerCredential
        }]
      },
    },
    format: {
      codec: 'H264',
        resolution: {
            width: 1920,
            height: 1080
        },
        crop:{
            topLeft: {
                x: videoCropTopLeftX,
                y: 0
            },
            bottomRight: {
                x: videoCropBottomRightX,
                y: 1080
            }
        },
        bitrate: 2000000
    },
    talkingAvatar: {
      character: TalkingAvatarCharacter,
      style: TalkingAvatarStyle,
      background: {
          color: backgroundColor
      }
  }
  }

  // Callback function to handle the response from TTS Avatar API
  const complete_cb = function (result) {
    const sdp = result.properties.getProperty(SpeechSDK.PropertyId.TalkingAvatarService_WebRTC_SDP)
    if (sdp === undefined) {
      console.log("Failed to get remote SDP. The avatar instance is temporarily unavailable. Result ID: " + result.resultId)
      document.getElementById('startSession').disabled = false
    }

    peerConnection.setRemoteDescription(new RTCSessionDescription(JSON.parse(atob(sdp)))).then(r => { })
  }

  const error_cb = function (result) {
    let cancellationDetails = SpeechSDK.CancellationDetails.fromResult(result)
    console.log(cancellationDetails)
    document.getElementById('startSession').disabled = false
  }

  // Call TTS Avatar API
  speechSynthesizer.setupTalkingAvatarAsync(JSON.stringify(clientRequest), complete_cb, error_cb)
}

window.startSession = () => {
  // Create the <i> element
  var iconElement = document.createElement("i");
  iconElement.className = "fa fa-spinner fa-spin";
  iconElement.id = "loadingIcon"
  var parentElement = document.getElementById("playVideo");
  parentElement.prepend(iconElement);

  speechSynthesisConfig.speechSynthesisVoiceName = TTSVoice
  document.getElementById('playVideo').className = "round-button-hide"

  fetch("/api/getSpeechToken", {
    method: "POST"
  })
    .then(response => response.text())
    .then(response => { 
      speechSynthesisConfig.authorizationToken = response;
      token = response
    })
    .then(() => {
      speechSynthesizer = new SpeechSDK.SpeechSynthesizer(speechSynthesisConfig, null)
      requestAnimationFrame(setupWebRTC)
    })

  
  // setupWebRTC()
}

async function greeting() {
  addToConversationHistory("Hello, my name is Lisa. How can I help you?", "light")

  let spokenText = "<speak version='1.0' xmlns='http://www.w3.org/2001/10/synthesis' xmlns:mstts='https://www.w3.org/2001/mstts' xml:lang='en-US'><voice xml:lang='en-US' xml:gender='Female' name='en-US-JennyNeural'>Hello, my name is Lisa. How can I help you?</voice></speak>"
  speechSynthesizer.speakSsmlAsync(spokenText, (result) => {
    if (result.reason === SpeechSDK.ResultReason.SynthesizingAudioCompleted) {
      console.log("Speech synthesized to speaker for text [ " + spokenText + " ]. Result ID: " + result.resultId)
    } else {
      console.log("Unable to speak text. Result ID: " + result.resultId)
      if (result.reason === SpeechSDK.ResultReason.Canceled) {
        let cancellationDetails = SpeechSDK.CancellationDetails.fromResult(result)
        console.log(cancellationDetails.reason)
        if (cancellationDetails.reason === SpeechSDK.CancellationReason.Error) {
          console.log(cancellationDetails.errorDetails)
        }
      }
    }
  })
}

window.speak = (text) => {
  async function speak(text) {
    addToConversationHistory(text, 'dark')

    fetch("/api/detectLanguage?text="+text, {
      method: "POST"
    })
      .then(response => response.text())
      .then(async language => {
        console.log(`Detected language: ${language}`);
        console.log(`Sending this input to the generateText function: ${text}`);

        const generatedResult = await generateText(text);

        console.log(`Called generatetext function sucessfully`);
        
        let spokenTextssml = `<speak version='1.0' xmlns='http://www.w3.org/2001/10/synthesis' xmlns:mstts='https://www.w3.org/2001/mstts' xml:lang='en-US'><voice xml:lang='en-US' xml:gender='Female' name='en-US-JennyMultilingualNeural'><lang xml:lang="${language}">${generatedResult}</lang></voice></speak>`

        if (language == 'ar-AE') {
          spokenTextssml = `<speak version='1.0' xmlns='http://www.w3.org/2001/10/synthesis' xmlns:mstts='https://www.w3.org/2001/mstts' xml:lang='en-US'><voice xml:lang='en-US' xml:gender='Female' name='ar-AE-FatimaNeural'><lang xml:lang="${language}">${generatedResult}</lang></voice></speak>`
        }
        let spokenText = generatedResult
        speechSynthesizer.speakSsmlAsync(spokenTextssml, (result) => {
          if (result.reason === SpeechSDK.ResultReason.SynthesizingAudioCompleted) {
            console.log("Speech synthesized to speaker for text [ " + spokenText + " ]. Result ID: " + result.resultId)
          } else {
            console.log("Unable to speak text. Result ID: " + result.resultId)
            if (result.reason === SpeechSDK.ResultReason.Canceled) {
              let cancellationDetails = SpeechSDK.CancellationDetails.fromResult(result)
              console.log(cancellationDetails.reason)
              if (cancellationDetails.reason === SpeechSDK.CancellationReason.Error) {
                console.log(cancellationDetails.errorDetails)
              }
            }
          }
        })
      })
      .catch(error => {
        console.error('Error:', error);
      });
  }
  speak(text);
}

window.stopSession = () => {
  speechSynthesizer.close()
}

window.startRecording = () => {
  const speechConfig = SpeechSDK.SpeechConfig.fromAuthorizationToken(token, 'eastus');
  speechConfig.authorizationToken = token;
  speechConfig.SpeechServiceConnection_LanguageIdMode = "Continuous";
  var autoDetectSourceLanguageConfig = SpeechSDK.AutoDetectSourceLanguageConfig.fromLanguages(supported_languages);
  // var autoDetectSourceLanguageConfig = SpeechSDK.AutoDetectSourceLanguageConfig.fromLanguages(["en-US"]);

  document.getElementById('buttonIcon').className = "fas fa-stop"
  document.getElementById('startRecording').disabled = true

  recognizer = SpeechSDK.SpeechRecognizer.FromConfig(speechConfig, autoDetectSourceLanguageConfig);

  recognizer.recognized = function (s, e) {
    if (e.result.reason === SpeechSDK.ResultReason.RecognizedSpeech) {
      console.log('Recognized:', e.result.text);
      window.stopRecording();
      // TODO: append to conversation
      window.speak(e.result.text);
    }
  };

  recognizer.startContinuousRecognitionAsync();

  console.log('Recording started.');
}

window.stopRecording = () => {
  if (recognizer) {
    recognizer.stopContinuousRecognitionAsync(
      function () {
        recognizer.close();
        recognizer = undefined;
        document.getElementById('buttonIcon').className = "fas fa-microphone"
        document.getElementById('startRecording').disabled = false
        console.log('Recording stopped.');
      },
      function (err) {
        console.error('Error stopping recording:', err);
      }
    );
  }
}

window.submitText = () => {
  document.getElementById('spokenText').textContent = document.getElementById('textinput').currentValue
  document.getElementById('textinput').currentValue = ""
  window.speak(document.getElementById('textinput').currentValue);
}


function addToConversationHistory(item, historytype) {
  const list = document.getElementById('chathistory');
  const newItem = document.createElement('li');
  newItem.classList.add('message');
  newItem.classList.add(`message--${historytype}`);
  newItem.textContent = item;
  list.appendChild(newItem);
}

function addProductToChatHistory(product) {
  const list = document.getElementById('chathistory');
  const listItem = document.createElement('li');
  listItem.classList.add('product');
  listItem.innerHTML = `
    <fluent-card class="product-card">
      <div class="product-card__header">
        <img src="${product.image_url}" alt="tent" width="100%">
      </div>
      <div class="product-card__content">
        <div><span class="product-card__price">$${product.special_offer}</span> <span class="product-card__old-price">$${product.original_price}</span></div>
        <div>${product.tagline}</div>
      </div>
    </fluent-card>
  `;
  list.appendChild(listItem);
}

// Make video background transparent by matting
function makeBackgroundTransparent(timestamp) {
  // Throttle the frame rate to 30 FPS to reduce CPU usage
  if (timestamp - previousAnimationFrameTimestamp > 30) {
      video = document.getElementById('video')
      tmpCanvas = document.getElementById('tmpCanvas')
      tmpCanvasContext = tmpCanvas.getContext('2d', { willReadFrequently: true })
      tmpCanvasContext.drawImage(video, 0, 0, video.videoWidth, video.videoHeight)
      if (video.videoWidth > 0) {
          let frame = tmpCanvasContext.getImageData(0, 0, video.videoWidth, video.videoHeight)
          for (let i = 0; i < frame.data.length / 4; i++) {
              let r = frame.data[i * 4 + 0]
              let g = frame.data[i * 4 + 1]
              let b = frame.data[i * 4 + 2]
              
              if (g - 150 > r + b) {
                  // Set alpha to 0 for pixels that are close to green
                  frame.data[i * 4 + 3] = 0
              } else if (g + g > r + b) {
                  // Reduce green part of the green pixels to avoid green edge issue
                  adjustment = (g - (r + b) / 2) / 3
                  r += adjustment
                  g -= adjustment * 2
                  b += adjustment
                  frame.data[i * 4 + 0] = r
                  frame.data[i * 4 + 1] = g
                  frame.data[i * 4 + 2] = b
                  // Reduce alpha part for green pixels to make the edge smoother
                  a = Math.max(0, 255 - adjustment * 4)
                  frame.data[i * 4 + 3] = a
              }
          }

          canvas = document.getElementById('canvas')
          canvasContext = canvas.getContext('2d')
          canvasContext.putImageData(frame, 0, 0);
      }

      previousAnimationFrameTimestamp = timestamp
  }

  window.requestAnimationFrame(makeBackgroundTransparent)
}