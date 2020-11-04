function loadImage(e){
    e.preventDefault();
    console.log("Add this!");
    /*
    $.ajax({
    type: "GET",
    data: {
        'XValue': $('#XValue').val(),
        'YValue':$('#YValue').val()
    },
    success: function(response){
        $.get("/?ReturnData=True", function(data){
            $('body').html(data);
        });
    }
    });
    */

    var form= new FormData($('#upload_image')[0]);
    
   $.ajax({
    url: "/imageUpload",
    data: form,
    cache: false,
    type: 'POST',
    contentType: false, // NEEDED, DON'T OMIT THIS (requires jQuery 1.6+)
    processData: false, // NEEDED, DON'T OMIT THIS
    // ... Other options like success and etc
    success: function(response){
        console.log('Pic was sent to server!');
        console.log(response.pred1.toString());
        $("#pred1").text(response.pred1.toString());
        $("#pred2").text(response.pred2.toString());
        $("#pred3").text(response.pred3.toString());
    }
    });

}

  

function previewPhoto(e) {
    frame.src=URL.createObjectURL(e.target.files[0]);
}