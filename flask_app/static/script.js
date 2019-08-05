$(function() {
    $('button#process').on('click', function() {
      $.getJSON($SCRIPT_ROOT + '/_input_helper', {
        text_data: $('#text_data').val()
      }, function(data) {
        $("#result").html(data.result);
      });
      return false;
    });
  });