// document.addEventListener("DOMContentLoaded", function() {
window.onload = function() {

    var socket = io();
   
    socket.on('connect', function() {
        console.log('Connected to the server');
    });
    
    socket.on('end_experiment', function(data) {
        window.location.href = "/end_experiment?reason=" + encodeURIComponent(data.reason);
    });
    
    socket.on('error_page', function(data) {
        window.location.href = "/error_page?reason=" + encodeURIComponent(data.reason);
    });
    
    socket.on('initiate_disconnect', function() {
        socket.disconnect();
    });
    
};

// window.onload = function() {
    //     var socket = io();
    // });