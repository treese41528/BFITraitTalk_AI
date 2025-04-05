// static/js/interview.js

document.addEventListener('DOMContentLoaded', () => {
    // DOM Elements
    const chatMessages = document.getElementById('chatMessages');
    const typingIndicator = document.getElementById('typingIndicator');
    const userInput = document.getElementById('userInput');
    const sendButton = document.getElementById('sendButton');
    const chatForm = document.getElementById('chatForm');
    const startButton = document.getElementById('startButton');
    const resetButton = document.getElementById('resetButton');
    const progressBar = document.getElementById('progressBar');
    const questionsContainer = document.getElementById('questionsContainer');
    const resultsSection = document.getElementById('resultsSection');
    const traitResults = document.getElementById('traitResults');
    
    // State variables
    let currentQuestionId = null;
    let isInterviewCompleted = false;
    
    // Initialize scale option click handlers
    initializeScaleOptions();
    
    // Start button click handler
    startButton.addEventListener('click', startInterview);
    
    // Reset button click handler
    resetButton.addEventListener('click', resetInterview);
    
    // Chat form submit handler
    chatForm.addEventListener('submit', handleChatSubmit);
    
    /**
     * Initialize event handlers for the Likert scale options
     */
    function initializeScaleOptions() {
        const scaleOptions = document.querySelectorAll('.scale-option');
        
        scaleOptions.forEach(option => {
            option.addEventListener('click', function() {
                const questionId = this.dataset.questionId;
                const value = this.dataset.value;
                const questionElement = document.getElementById(`question-${questionId}`);
                
                // Check if this question is the current one
                if (questionElement.classList.contains('current')) {
                    // Remove selected class from all options in this question
                    const options = questionElement.querySelectorAll('.scale-option');
                    options.forEach(opt => opt.classList.remove('selected'));
                    
                    // Add selected class to clicked option
                    this.classList.add('selected');
                    
                    // Update the answer status
                    const statusElement = document.getElementById(`status-${questionId}`);
                    if (value === 'skipped') {
                        statusElement.textContent = 'Skipped';
                        questionElement.classList.add('skipped');
                    } else {
                        statusElement.textContent = `Answered: ${value}`;
                        questionElement.classList.add('answered');
                        questionElement.classList.remove('skipped');
                    }
                    
                    // Save the answer
                    saveAnswer(questionId, value);
                    
                    // Auto-send the answer via chat
                    let message = '';
                    if (value === 'skipped') {
                        message = "I'd prefer to skip this question.";
                    } else {
                        const scaleLabels = ['', 'Strongly Disagree', 'Disagree', 'Neutral', 'Agree', 'Strongly Agree'];
                        message = `I'd rate this as ${value} - ${scaleLabels[value]}.`;
                    }
                    
                    userInput.value = message;
                    sendMessage();
                }
            });
        });
    }
    
    /**
     * Start the interview process
     */
    function startInterview() {
        showTypingIndicator();
        
        // Disable start button
        startButton.disabled = true;
        
        fetch('/api/start', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            }
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                // Hide start button, show reset button
                startButton.style.display = 'none';
                resetButton.style.display = 'inline-block';
                
                // Display AI greeting
                appendMessage('model', data.ai_message);
                
                // Enable user input
                userInput.disabled = false;
                sendButton.disabled = false;
                userInput.focus();
                
                // Highlight current question if available
                if (data.current_question_id) {
                    highlightQuestion(data.current_question_id);
                }
            } else {
                appendMessage('system', 'Failed to start interview. Please try again.');
                startButton.disabled = false;
            }
            
            hideTypingIndicator();
        })
        .catch(error => {
            console.error('Error starting interview:', error);
            appendMessage('system', 'An error occurred while starting the interview.');
            startButton.disabled = false;
            hideTypingIndicator();
        });
    }
    
    /**
     * Reset the interview
     */
    function resetInterview() {
        if (confirm('Are you sure you want to reset the interview? All progress will be lost.')) {
            // Clear session
            fetch('/clear_session')
                .then(() => {
                    // Reload the page
                    window.location.reload();
                })
                .catch(error => {
                    console.error('Error resetting interview:', error);
                    alert('Failed to reset interview. Please refresh the page manually.');
                });
        }
    }
    
    /**
     * Handle chat form submission
     */
    function handleChatSubmit(event) {
        event.preventDefault();
        
        if (userInput.value.trim() === '') {
            return; // Don't send empty messages
        }
        
        sendMessage();
    }
    
    /**
     * Send user message to the server
     */
    function sendMessage() {
        const message = userInput.value.trim();
        
        // Append user message to chat
        appendMessage('user', message);
        
        // Clear input and disable input temporarily
        userInput.value = '';
        userInput.disabled = true;
        sendButton.disabled = true;
        
        // Show typing indicator
        showTypingIndicator();
        
        // Send message to server
        fetch('/api/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ message: message })
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                // Append AI response
                appendMessage('model', data.ai_message);
                console.log("Received data:", data);
                // Update form if needed
                if (data.form_update) {
                    updateForm(data.form_update.question_id, data.form_update.answer);
                }
                
                // Update progress bar
                if (data.progress) {
                    updateProgress(data.progress);
                }
                
                // Check if interview is completed
                if (data.is_completed) {
                    completeInterview();
                } else {
                    console.log('Highlighting next question ID:', data.current_question_id);
                    // Highlight next question if available
                    if (data.current_question_id) {
                        highlightQuestion(data.current_question_id);
                    }
                    
                    // Re-enable input for next message
                    userInput.disabled = false;
                    sendButton.disabled = false;
                    userInput.focus();
                }
            } else {
                appendMessage('system', data.error || 'An error occurred. Please try again.');
                userInput.disabled = false;
                sendButton.disabled = false;
            }
            
            hideTypingIndicator();
        })
        .catch(error => {
            console.error('Error sending message:', error);
            appendMessage('system', 'An error occurred while sending your message.');
            userInput.disabled = false;
            sendButton.disabled = false;
            hideTypingIndicator();
        });
    }
    
    /**
     * Append a message to the chat
     */
    function appendMessage(role, text) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${role}`;
        
        // Format text (convert newlines to <br>)
        messageDiv.innerHTML = text.replace(/\n/g, '<br>');
        
        // Insert before typing indicator
        chatMessages.insertBefore(messageDiv, typingIndicator);
        
        // Scroll to bottom
        scrollToBottom();
    }
    
    /**
     * Show typing indicator
     */
    function showTypingIndicator() {
        typingIndicator.style.display = 'flex';
        scrollToBottom();
    }
    
    /**
     * Hide typing indicator
     */
    function hideTypingIndicator() {
        typingIndicator.style.display = 'none';
    }
    
    /**
     * Scroll chat to bottom
     */
    function scrollToBottom() {
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }
    
    /**
     * Update the form with an answer
     */
    function updateForm(questionId, answer) {
        const questionElement = document.getElementById(`question-${questionId}`);
        if (!questionElement) return;
        
        // Remove current highlight
        questionElement.classList.remove('current');
        
        // Add answered class
        if (answer === 'skipped') {
            questionElement.classList.add('answered', 'skipped');
        } else {
            questionElement.classList.add('answered');
        }
        
        // Update the answer status
        const statusElement = document.getElementById(`status-${questionId}`);
        if (statusElement) {
            if (answer === 'skipped') {
                statusElement.textContent = 'Skipped';
            } else {
                const scaleLabels = ['', 'Strongly Disagree', 'Disagree', 'Neutral', 'Agree', 'Strongly Agree'];
                statusElement.textContent = `Answered: ${answer} - ${scaleLabels[answer]}`;
            }
        }
        
        // Select the correct option
        const options = questionElement.querySelectorAll('.scale-option');
        options.forEach(option => {
            option.classList.remove('selected');
            if (option.dataset.value == answer) {
                option.classList.add('selected');
            }
        });
    }
    
    /**
     * Highlight a question
     */
    function highlightQuestion(questionId) {
        // *** ADD CONSOLE LOGS HERE ***
        console.log(`Attempting to highlight QID: ${questionId}`);
        console.log(`Current highlighted was: ${currentQuestionId}`);
        // *** END CONSOLE LOGS ***

        // Remove highlight from previous question
        if (currentQuestionId) {
            const previousQuestion = document.getElementById(`question-${currentQuestionId}`);
            if (previousQuestion) {
                console.log(`Removing .current from #question-${currentQuestionId}`);
                previousQuestion.classList.remove('current');
            } else {
                console.warn(`Previous question element #question-${currentQuestionId} not found.`);
            }
        }

        // Add highlight to new question
        currentQuestionId = questionId; // Update the global tracker
        const questionElement = document.getElementById(`question-${questionId}`);

        if (questionElement) {
            console.log(`Adding .current to #question-${questionId}`);
            questionElement.classList.add('current');

            // Scroll to question
            console.log(`Scrolling to #question-${questionId}`);
            questionElement.scrollIntoView({ behavior: 'smooth', block: 'center' });
        } else {
            console.error(`Cannot highlight: Question element #question-${questionId} not found.`);
            currentQuestionId = null; // Reset tracker if element not found
        }
    }
    
    /**
     * Save an answer via API
     */
    function saveAnswer(questionId, answer) {
        fetch('/api/save_answer', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                question_id: questionId,
                answer_value: answer
            })
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                // If next question is available, highlight it
                if (data.current_question_id) {
                    highlightQuestion(data.current_question_id);
                }
            } else {
                console.error('Error saving answer:', data.error);
            }
        })
        .catch(error => {
            console.error('Error saving answer:', error);
        });
    }
    
    /**
     * Update progress bar
     */
    function updateProgress(percentage) {
        progressBar.style.width = `${percentage}%`;
        progressBar.textContent = `${percentage}%`;
        progressBar.setAttribute('data-progress', percentage);
    }
    
    /**
     * Complete the interview and show results
     */
    function completeInterview() {
        isInterviewCompleted = true;
        
        // Disable input
        userInput.disabled = true;
        sendButton.disabled = true;
        
        appendMessage('system', 'Interview completed! Generating your personality profile...');
        showTypingIndicator();
        
        // Fetch results
        fetch('/api/results')
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    // Display results
                    displayResults(data.report);
                    appendMessage('model', 'Your personality profile is ready! You can review your results in the form panel.');
                } else {
                    appendMessage('system', 'Failed to generate results. Please try again.');
                }
                
                hideTypingIndicator();
            })
            .catch(error => {
                console.error('Error getting results:', error);
                appendMessage('system', 'An error occurred while generating results.');
                hideTypingIndicator();
            });
    }
    
    /**
     * Display results in the form panel
     */
    function displayResults(report) {
        // Clear previous results
        traitResults.innerHTML = '';
        
        // Create results for each trait
        const traits = ['openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism'];
        
        traits.forEach(trait => {
            if (report.traits[trait]) {
                const traitData = report.traits[trait];
                const interpretation = report.interpretations[trait];
                
                const traitElement = document.createElement('div');
                traitElement.className = 'trait-result';
                
                // Calculate percentage for bar
                const percentage = Math.min(100, Math.round((traitData.score / 5) * 100));
                
                traitElement.innerHTML = `
                    <div class="trait-header">
                        <span class="trait-name">${trait.charAt(0).toUpperCase() + trait.slice(1)}</span>
                        <span class="trait-score">${traitData.score.toFixed(1)}</span>
                    </div>
                    <div class="trait-bar">
                        <div class="trait-bar-fill ${trait}" style="width: ${percentage}%"></div>
                    </div>
                    <div class="trait-description">
                        ${interpretation.interpretation}
                    </div>
                `;
                
                traitResults.appendChild(traitElement);
            }
        });
        
        // Show results section
        resultsSection.style.display = 'block';
        resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
});