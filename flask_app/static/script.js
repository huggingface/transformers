$(function() {
    $('button#process').on('click', function() {
      $.getJSON($SCRIPT_ROOT + '/_input_helper', {
        text_data: $('#text_data').val(),
        question_data: $('#question_data').val()
      }, function(data) {
        $("#result").html(data.result);
      });
      return false;
    });
  });

$(function() {
    $('button#random-btn').on('click', function() {
      $.getJSON($SCRIPT_ROOT + '/_random_page', {
      }, function(data) {
        $("#text_data").val(data.context);
        $("#question_data").val(data.question);
      });
      return false;
    });
});