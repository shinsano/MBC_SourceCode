const lowercaseKeys = obj =>
  Object.keys(obj).reduce((acc, key) => {
    acc[key.toLowerCase()] = obj[key];
    return acc;
  }, {});
  
function DragNSort(config) {
  this.$activeItem = null;
  this.$container = config.container;
  this.$items = this.$container.querySelectorAll('.' + config.itemClass);
  this.dragStartClass = config.dragStartClass;
  this.dragEnterClass = config.dragEnterClass;
}

DragNSort.prototype.removeClasses = function () {
  [].forEach.call(this.$items, function ($item) {
    $item.classList.remove(this.dragStartClass, this.dragEnterClass);
  }.bind(this));
};

DragNSort.prototype.on = function (elements, eventType, handler) {
  [].forEach.call(elements, function (element) {
    element.addEventListener(eventType, handler.bind(element, this), false);
  }.bind(this));
};

DragNSort.prototype.onDragStart = function (_this, event) {
  _this.$activeItem = this;

  this.classList.add(_this.dragStartClass);
  event.dataTransfer.effectAllowed = 'move';
  event.dataTransfer.setData('text/html', this.innerHTML);
};

DragNSort.prototype.onDragEnd = function (_this) {
  this.classList.remove(_this.dragStartClass);
};

DragNSort.prototype.onDragEnter = function (_this) {
  this.classList.add(_this.dragEnterClass);
};

DragNSort.prototype.onDragLeave = function (_this) {
  this.classList.remove(_this.dragEnterClass);
};

DragNSort.prototype.onDragOver = function (_this, event) {
  if (event.preventDefault) {
    event.preventDefault();
  }

  event.dataTransfer.dropEffect = 'move';

  return false;
};

DragNSort.prototype.onDrop = function (_this, event) {
  if (event.stopPropagation) {
    event.stopPropagation();
  }

  if (_this.$activeItem !== this) {
    target_index = parseInt(event.target.id.replace('img', ''));
    source_index = parseInt(_this.$activeItem.id.replace('img', ''));
    var tmp_image = images[target_index - 1];
    var tmp_label = labels[target_index - 1];
    images[target_index - 1] = images[source_index - 1];
    labels[target_index - 1] = labels[source_index - 1];
    images[source_index - 1] = tmp_image;
    labels[source_index - 1] = tmp_label;
    _this.$activeItem.innerHTML = this.innerHTML;
    this.innerHTML = event.dataTransfer.getData('text/html');
  }

  _this.removeClasses();

  return false;
};

DragNSort.prototype.bind = function () {
  this.on(this.$items, 'dragstart', this.onDragStart);
  this.on(this.$items, 'dragend', this.onDragEnd);
  this.on(this.$items, 'dragover', this.onDragOver);
  this.on(this.$items, 'dragenter', this.onDragEnter);
  this.on(this.$items, 'dragleave', this.onDragLeave);
  this.on(this.$items, 'drop', this.onDrop);
};

DragNSort.prototype.init = function () {
  this.bind();
};

function uuidv4() {
  return ([1e7]+'').replace(/[01]/g, c =>
    (c ^ crypto.getRandomValues(new Uint8Array(1))[0] & 15 >> c / 4).toString(16)
  );
}

var userID = uuidv4();
var iterationID = 0;

var images = [null, null, null, null, null, null, null, null, null];
var labels = [null, null, null, null, null, null, null, null, null];
var deleted_images = [];
var vision_api_key = "AIzaSyA6BW54moCxhWMKoAxbRejAsCHdSUdYK0Y";
var option1 = [
[1.6, 1.6],
[1.6, 1.4],
[1.4, 1.6],
[1.4, 1.4],
[1.6, 1],
[1, 1.6],
[1.4, 1],
[1, 1.4],
[1, 1]
];

var option2 = [
[3.4, 3.4],
[3.4, 2.6],
[2.6, 3.4],
[2.6, 2.6],
[3.4, 0.6],
[0.6, 3.4],
[2.6, 0.6],
[0.6, 2.6],
[0.6, 0.6]
];
var weights = option2;
var pending_request = 0;
var temp_images = [];
var temp_labels = [];
var word1 = "";
var word2 = "";

$("#startBtn").on("click", function (e) {
  word1 = $("#word1").val();
  word2 = $("#word2").val();
  if (word1 != "" && word2 != "") {
    $('#loading').css('display', 'block');
    search(word1 + '%20' + word2, false);
    $('#word1').prop('disabled', true)
    $('#word2').prop('disabled', true)
    $('#word1_text').text(word1)
    $('#word2_text').text(word2)
    // $('#options').css('visibility', 'initial');
    $('#resetBtn').css('visibility', 'initial');
    $('#nextBtn').css('visibility', 'initial');
    $('#exportBtn').css('visibility', 'initial');
    e.target.hidden = true;
  } else {
    $.toast({
      heading: 'Error',
      text: 'Please input all words for start',
      showHideTransition: 'fade',
      position: 'bottom-left',
      icon: 'error'
    })
  }
});

$('.drag-item').on("click", '.icon-delete', function(e) {
  var deleted_src = $(e.currentTarget.parentNode).find('img')[0]?.src
  var delete_index = parseInt(e.currentTarget.parentNode.id.replace('img', ''));
  images[delete_index - 1] = null;
  labels[delete_index - 1] = null;
  if (deleted_src !== undefined && !deleted_images.includes(deleted_src)) {
    deleted_images.push(deleted_src);
  }
  $(e.currentTarget.parentNode).find('img')[0]?.remove();
});

function search(query, is_next) {
  if (is_next)
    pending_request++;
  $.getJSON("/api/search_new?word=" + query,
    function (data) {
      if (is_next) {
        var non_empty_data = (data[0].length > 0) ? data[0] : (data[1].length > 0) ? data[1] : (data[2].length > 0) ? data[2] : null;
        if (non_empty_data != null) {
          non_empty_data = non_empty_data.filter(n => !deleted_images.includes(n));
        }
        if (non_empty_data != null && non_empty_data.length !== 0) {
          insertOne(non_empty_data);
          pending_request--;
        } else {
          pending_request--;
          check_finish();
        }
      } else {
        fill_empty_image(data);
      }
    });
}

function fill_empty_image(data) {
  for (item in data) {
    data[item] = data[item].filter(n => !deleted_images.includes(n));
  }
  var remain_count = images.filter(item => item == null).length;
  var tmp_stack = []
  var inserted_count = 0;
  while (inserted_count != remain_count) {
    var new_image = data[inserted_count % 3][Math.floor(inserted_count / 3)];
    if (!tmp_stack.includes(new_image)) {
      tmp_stack.push(new_image);
      queryImageLabels(new_image);
    } else {
      remain_count++;
    }
    inserted_count++;    
  }
}

function insertOne(non_empty_data) {
    var new_image = ''
    if (non_empty_data.length > 9)
      new_image = non_empty_data[Math.round(Math.random() * 9)];
    else
      new_image = non_empty_data[Math.round(Math.random() * (non_empty_data.length - 1))];
    queryImageLabels(new_image);
}

function queryImageLabels(url) {
  pending_request++;
  var json = '{' +
    ' "requests": [' +
    '   { ' +
    '     "image": {' +
    '       "source": {' +
    '         "imageUri": "' + url + '"' +
    '       },' +
    '     },' +
    '     "features": [' +
    '         {' +
    '           "type": "LABEL_DETECTION",' +
    '           "maxResults": 5' +
    '         }' +
    '     ]' +
    '   }' +
    ']' +
    '}';

  $.ajax({
    type: 'POST',
    url: "https://vision.googleapis.com/v1/images:annotate?key=" + vision_api_key,
    dataType: 'json',
    data: json,
    headers: {
      "Content-Type": "application/json",
    },
    success: function (data, textStatus, jqXHR) {
      labelObj = {};      
      try {
        for (i = 0; i < data['responses'][0]['labelAnnotations'].length; i++) {
          var term = data['responses'][0]['labelAnnotations'][i]['description'];
          var score = data['responses'][0]['labelAnnotations'][i]['score'];
          if (!Object.keys(labelObj).includes(term)) {
            labelObj[term] = score;
          }
        }
  
        labelObj = lowercaseKeys(labelObj);
        var original_keys = Object.keys(labelObj)
        for (var key_index in original_keys) {
          if (original_keys[key_index].indexOf(' ') != -1) {
              var score = labelObj[original_keys[key_index]]
              delete labelObj[original_keys[key_index]]
              original_keys[key_index] = original_keys[key_index].replace(' and ', ' ').replace(' & ', ' ')
              var subkeys = original_keys[key_index].split(' ')
              for (const sub_index in subkeys)
                if (!(subkeys[sub_index] in labelObj))
                  labelObj[subkeys[sub_index]] = score
          }
        }
        temp_images.push(url);
        temp_labels.push(labelObj);
      } catch (error) {
        console.log(error);        
      }      
      pending_request--;
      check_finish();
    },
    error: function () {
      pending_request--;
      check_finish();
    }
  });

}

function check_finish() {
  if (pending_request == 0) {
    for (i in images) {
      if (images[parseInt(i)] == null && temp_images.length > 0) {
        while(temp_images.length > 0) {
          var selected_index = Math.round(Math.random() * (temp_images.length - 1))
          if (!images.includes(temp_images[selected_index])) {
            images[parseInt(i)] = temp_images[selected_index];
            labels[parseInt(i)] = temp_labels[selected_index];
            temp_images.splice(selected_index, 1);
            temp_labels.splice(selected_index, 1);
            $('#img' + (parseInt(i) + 1)).find('img')[0]?.remove()
            $('#img' + (parseInt(i) + 1)).append('<img src="' + images[parseInt(i)] + '" alt="" class="w-100 h-100">')
            break;
          } else {
            temp_images.splice(selected_index, 1);
            temp_labels.splice(selected_index, 1);
          }
        }        
      }
    }
    $('#loading').css('display', 'none');
    console.log('finished');
    temp_images = [];
    temp_labels = [];
  }
}

function generateLogData() {
  var logdata = [
    userID,
    word1,
    word2,
    iterationID,
    labels.map(a => a ? Object.keys(a).join(','): '-').join('\n'),
    labels.map(a => a ? Object.values(a).join(','): '-').join('\n'),
    '','','','','','','','','',
    '','','','','','','','','',
    '','',
    '',
    Date()
  ]
  
  return logdata
}

$('#nextBtn').on("click", function(e) {
  iterationID++;
  var logData = generateLogData();
  $('#loading').css('display', 'block');
  req_labels = []
  for (label_index in labels) {
    if (labels[label_index] != null) {
      req_labels.push({
        image_weight: weights[label_index],
        image_labels: labels[label_index],
        label_index: label_index
      })
    }
  }
  
  $.ajax({
      url: "/api/next_search_new2",
      type: "POST",
      data: JSON.stringify({ labels: req_labels, word1: word1, word2: word2, log: logData, feedbacks: [], sheet: 1 }),
      contentType: "application/json",
      dataType: "json",
      cache: false,
      success: function (data) {
        data.forEach((item) => search(item[0], true));
      }
  });
});

$('#resetBtn').on("click", function(e) {
  images = [null, null, null, null, null, null, null, null, null];
  labels = [null, null, null, null, null, null, null, null, null];
  pending_request = 0;
  temp_images = [];
  temp_labels = [];
  for (const i of Array(9).keys()) {
    $('#img' + (parseInt(i) + 1)).find('img')[0]?.remove()
  }
  $('#word1').val('')
  $('#word2').val('')
  $('#word1_text').text('Word1')
  $('#word2_text').text('Word2')
  // $('#options').css('visibility', 'hidden');
  $('#word1').prop('disabled', false)
  $('#word2').prop('disabled', false)
  $('#startBtn').prop('hidden', false);
  $('#resetBtn').css('visibility', 'hidden');
  $('#nextBtn').css('visibility', 'hidden');
  $('#exportBtn').css('visibility', 'hidden');
});

// $('#select_option').on("change", function(e) {
//   if (e.target.checked) {
//     weights = option2;
//   } else {
//     weights = option1;
//   }
// });

$('#exportBtn').on("click", function(e) {
  html2canvas(document.getElementById('export'), {
    backgroundColor: '#343a40',
    useCORS: true
  })
    .then(function (canvas) {
        theCanvas = canvas;
  
        canvas.toBlob(function (blob) {
            saveAs(blob, "Moodboard_" + userID + ".png");
        });
    })
    .catch(function (err) {
        console.log(err);
    });
});

function init() {
  // Instantiate
  var draggable = new DragNSort({
    container: document.querySelector('.drag-list'),
    itemClass: 'drag-item',
    dragStartClass: 'drag-start',
    dragEnterClass: 'drag-enter'
  });

  draggable.init();
}

$(document).ready(init);
