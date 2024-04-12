
  (function() {
    var fn = function() {
      Bokeh.safely(function() {
        (function(root) {
          function embed_document(root) {
            
          var docs_json = '{"2e86b4e1-26d9-46eb-b46e-1b7a2d47612d":{"roots":{"references":[{"attributes":{"default_sort":"descending","editor":{"id":"3395","type":"StringEditor"},"field":"Parameters","formatter":{"id":"3394","type":"StringFormatter"},"sortable":false,"title":"Parameters","width":150},"id":"3387","type":"TableColumn"},{"attributes":{},"id":"3399","type":"StringEditor"},{"attributes":{},"id":"3392","type":"Selection"},{"attributes":{"source":{"id":"3386","type":"ColumnDataSource"}},"id":"3391","type":"CDSView"},{"attributes":{"callback":null,"data":{"LPI":["87.73 +/- 93662.35","08.70 +/- 44125.42"],"Parameters":["optimizer","batch_size"],"fANOVA":["70.93 +/- 36.37","15.23 +/- 25.92"]},"selected":{"id":"3392","type":"Selection"},"selection_policy":{"id":"3393","type":"UnionRenderers"}},"id":"3386","type":"ColumnDataSource"},{"attributes":{"default_sort":"descending","editor":{"id":"3397","type":"StringEditor"},"field":"fANOVA","formatter":{"id":"3396","type":"StringFormatter"},"title":"fANOVA","width":100},"id":"3388","type":"TableColumn"},{"attributes":{},"id":"3394","type":"StringFormatter"},{"attributes":{},"id":"3398","type":"StringFormatter"},{"attributes":{"columns":[{"id":"3387","type":"TableColumn"},{"id":"3388","type":"TableColumn"},{"id":"3389","type":"TableColumn"}],"height":80,"index_position":null,"source":{"id":"3386","type":"ColumnDataSource"},"view":{"id":"3391","type":"CDSView"}},"id":"3390","type":"DataTable"},{"attributes":{},"id":"3393","type":"UnionRenderers"},{"attributes":{},"id":"3396","type":"StringFormatter"},{"attributes":{},"id":"3397","type":"StringEditor"},{"attributes":{"default_sort":"descending","editor":{"id":"3399","type":"StringEditor"},"field":"LPI","formatter":{"id":"3398","type":"StringFormatter"},"title":"LPI","width":100},"id":"3389","type":"TableColumn"},{"attributes":{},"id":"3395","type":"StringEditor"}],"root_ids":["3390"]},"title":"Bokeh Application","version":"1.1.0"}}';
          var render_items = [{"docid":"2e86b4e1-26d9-46eb-b46e-1b7a2d47612d","roots":{"3390":"bf8ed684-02be-468a-adac-a5309ba56a17"}}];
          root.Bokeh.embed.embed_items(docs_json, render_items);
        
          }
          if (root.Bokeh !== undefined) {
            embed_document(root);
          } else {
            var attempts = 0;
            var timer = setInterval(function(root) {
              if (root.Bokeh !== undefined) {
                embed_document(root);
                clearInterval(timer);
              }
              attempts++;
              if (attempts > 100) {
                console.log("Bokeh: ERROR: Unable to run BokehJS code because BokehJS library is missing");
                clearInterval(timer);
              }
            }, 10, root)
          }
        })(window);
      });
    };
    if (document.readyState != "loading") fn();
    else document.addEventListener("DOMContentLoaded", fn);
  })();
