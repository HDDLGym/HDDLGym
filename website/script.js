let currentStep = 0;
let totalSteps = 0;

// Fetch and render JSON data
fetch("data.json")
   .then(response => response.json())
   .then(data => {
      // Combine belief hierarchies of agent 0 and agent 1
      const beliefHierarchy0 = data.beliefHierarchy0Array;
      const beliefHierarchy1 = data.beliefHierarchy1Array;

      totalSteps = data.actionArrayList.length;

      // Render the first timestep
      renderStep(data, beliefHierarchy0, beliefHierarchy1);

      // Add event listeners for Next and Previous buttons
      document.getElementById("next-btn").addEventListener("click", () => {
         if (currentStep < totalSteps - 1) {
            currentStep++;
            renderStep(data, beliefHierarchy0, beliefHierarchy1);
         }
      });

      document.getElementById("prev-btn").addEventListener("click", () => {
         if (currentStep > 0) {
            currentStep--;
            renderStep(data, beliefHierarchy0, beliefHierarchy1);
         }
      });
   })
   .catch(error => console.error("Error loading data:", error));

// Function to render data for the current step
function renderStep(data, beliefHierarchy0, beliefHierarchy1) {
   // Update timestep display
   document.getElementById("timestep").textContent = `Timestep: ${currentStep + 1} of ${totalSteps}`;

   // Clear containers for each agent
   clearContainer("actions-agent1");
   clearContainer("hierarchies-agent1");
   clearContainer("beliefs-agent1");
   clearContainer("actions-agent2");
   clearContainer("hierarchies-agent2");
   clearContainer("beliefs-agent2");

   // Render data for each agent at the current timestep
   renderActions(data.actionArrayList[currentStep], 'chef1', document.getElementById("actions-agent1"));
   renderHierarchies(data.hierarchiesArrayList[currentStep][0], document.getElementById("hierarchies-agent1"));
   renderBeliefs(beliefHierarchy1[currentStep], document.getElementById("beliefs-agent2"));

   renderActions(data.actionArrayList[currentStep], 'chef2', document.getElementById("actions-agent2"));
   renderHierarchies(data.hierarchiesArrayList[currentStep][1], document.getElementById("hierarchies-agent2"));
   renderBeliefs(beliefHierarchy0[currentStep], document.getElementById("beliefs-agent1"));
}

// Helper function to clear a container
function clearContainer(containerId) {
   const container = document.getElementById(containerId);
   while (container.firstChild) {
      container.removeChild(container.firstChild);
   }
}

// Function to render actions for a specific chef
function renderActions(actions, chefKey, container) {
   const actionText = actions[chefKey] || 'No actions at this timestep.';
   const p = document.createElement('p');
   p.innerHTML = highlightKeywords(actionText);
   container.appendChild(p);
}

function renderHierarchies(hierarchy, container) {

   if (hierarchy) {
      renderNestedArray(hierarchy, container);
   } else {
      const p = document.createElement('p');
      p.textContent = 'No hierarchies at this timestep.';
      container.appendChild(p);
   }
}


// Function to render beliefs
function renderBeliefs(beliefs, container) {
   if (beliefs && beliefs.length > 0) {
      renderNestedArray(beliefs, container);
   } else {
      container.textContent = 'No beliefs at this timestep.';
   }
}

// Function to render nested arrays as lists
function renderNestedArray(arr, container) {
   const list = document.createElement('ul');
   arr.forEach(item => {
      const listItem = document.createElement('li');
      if (Array.isArray(item)) {
         renderNestedArray(item, listItem);
      } else {
         listItem.innerHTML = highlightKeywords(item);
      }
      list.appendChild(listItem);
   });
   container.appendChild(list);
}

// Highlight specific keywords in the text
function highlightKeywords(text) {
   return text.replace(/\b(deliver|make-soup|interact|wait|add-ingredient|cook|chef1|chef2|onion|pot1|bowl-pile|delivery)\b/g, '<span class="highlight">$1</span>');
}