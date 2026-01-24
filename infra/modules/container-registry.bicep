// Azure Container Registry module

@description('Name of the container registry')
param name string

@description('Location for the registry')
param location string

@description('Tags to apply')
param tags object

resource acr 'Microsoft.ContainerRegistry/registries@2023-07-01' = {
  name: name
  location: location
  tags: tags
  sku: {
    name: 'Basic'  // ~$5/month
  }
  properties: {
    adminUserEnabled: true
  }
}

output name string = acr.name
output loginServer string = acr.properties.loginServer
output id string = acr.id
